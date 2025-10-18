import json
import argparse
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

import huggingface_hub
from transformers import AutoTokenizer
from datasets import load_from_disk


def get_lengths(dataset):
    # ! ORDER IS RANDOM AS MAP DOES NOT GUARANTEE PROCESSING IN ORDER (? CHECK THIS ?)

    model_paths = [c["model"] for c in dataset[0]["completions"]]
    model_to_tokenizer = {}
    for path in model_paths:
        model_to_tokenizer[path] = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    def _get_lengths(sample):
        for completion in sample["completions"]:
            tokenizer = model_to_tokenizer[completion["model"]]

            messages = completion["messages"]
            messages.append({"role": "assistant", "content": completion["response_text"]})
            messages_formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            messages_tokenized = tokenizer.encode(messages_formatted, add_special_tokens=True)

            completion["length"] = len(messages_tokenized)
        
        return sample

    return dataset.map(_get_lengths)

def calculate_statistics(dataset):
    models = [c["model"] for c in dataset[0]["completions"]]
    statistics = {}

    for model in models:
        statistics[model] = {
            "lengths": [],
            "scores": [],
            "num_best": 0,
            "num_worst": 0,
            "think_count": 0
        }

    for sample_idx, sample in tqdm(enumerate(dataset)):
        scores = [(c["model"], c["overall_score"]) for c in sample["completions"]]

        min_model, _ = min(scores, key=lambda x: x[1])
        max_model, _ = max(scores, key=lambda x: x[1])
        statistics[min_model]["num_worst"] += 1
        statistics[max_model]["num_best"] += 1

        for completion_idx, completion in enumerate(sample["completions"]):
            model = completion["model"]
            statistics[model]["lengths"].append(int(completion["length"]))
            statistics[model]["scores"].append(float(completion["overall_score"]))
            if "<think>" in completion["response_text"]:
                statistics[model]["think_count"] += 1

    for model, stats in statistics.items():
        lengths = np.array(stats["lengths"])
        scores = np.array(stats["scores"])

        stats["length_mean"] = float(np.mean(lengths)) if len(lengths) > 0 else 0.0
        stats["length_std"] = float(np.std(lengths)) if len(lengths) > 0 else 0.0
        stats["score_mean"] = float(np.mean(scores)) if len(scores) > 0 else 0.0
        stats["score_std"] = float(np.std(scores)) if len(scores) > 0 else 0.0

        # Add binned length counts
        stats["len_bin_1"] = int(np.sum((lengths > 0) & (lengths <= 1000)))
        stats["len_bin_2"] = int(np.sum((lengths > 1000) & (lengths <= 2000)))
        stats["len_bin_3"] = int(np.sum((lengths > 2000) & (lengths <= 3000)))
        stats["len_bin_4"] = int(np.sum((lengths > 3000) & (lengths <= 4000)))
        stats["len_bin_5"] = int(np.sum(lengths > 4000))

    return statistics

def save_dataset(dataset, path):
    dataset.save_to_disk(path)

    with open(os.path.join(path, "first_sample.json"), "w") as f:
        json.dump(dataset[0], f, indent=4)

    print(f"Dataset saved to {path}")

def generate_table(completion_statistics: dict, annotation_statistics: list[dict]):
    # Headers for completion stats
    headers = [
        "Model",
        "Length Mean",
        "Length Std",
        "#0-1000",
        "#1000-2000",
        "#2000-3000",
        "#3000-4000",
        "#>4000",
        r"#\<think>",
    ]

    # Add headers for each annotation model
    for annotator_name, _ in annotation_statistics:
        headers.append(f"**{annotator_name}**")
        headers.extend([
            "Mean Score",
            "Std Score",
            "# Best",
            "# Worst",
        ])

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "|-" + "-|-".join(["-" * len(h) for h in headers]) + "-|"
    table_lines = [header_line, separator_line]

    # Get all model names
    model_names = list(completion_statistics.keys())

    # Rows
    for model_name in model_names:
        row = [
            model_name.split("/")[-1],
            f"{completion_statistics[model_name]['length_mean']:.2f}",
            f"{completion_statistics[model_name]['length_std']:.2f}",
            str(completion_statistics[model_name]["len_bin_1"]),
            str(completion_statistics[model_name]["len_bin_2"]),
            str(completion_statistics[model_name]["len_bin_3"]),
            str(completion_statistics[model_name]["len_bin_4"]),
            str(completion_statistics[model_name]["len_bin_5"]),
            str(completion_statistics[model_name]["think_count"]),
        ]

        for _, stats_dict in annotation_statistics:
            row.append("")  # Empty cell for the annotator name column
            if model_name in stats_dict:
                stats = stats_dict[model_name]
                row.extend([
                    f"{stats['score_mean']:.2f}",
                    f"{stats['score_std']:.2f}",
                    str(stats["num_best"]),
                    str(stats["num_worst"]),
                ])
            else:
                row.extend(["N/A"] * 4)
        
        table_lines.append("| " + " | ".join(row) + " |")

    # Aggregated Row
    agg_row = ["**Aggregated**"]
    
    # Completion stats aggregation
    all_lengths = [l for model_stats in completion_statistics.values() for l in model_stats.get("lengths", [])]
    if all_lengths:
        agg_row.append(f"{np.mean(all_lengths):.2f}")
        agg_row.append(f"{np.std(all_lengths):.2f}")
    else:
        agg_row.extend(["0.00", "0.00"])

    agg_row.append(str(sum(s["len_bin_1"] for s in completion_statistics.values())))
    agg_row.append(str(sum(s["len_bin_2"] for s in completion_statistics.values())))
    agg_row.append(str(sum(s["len_bin_3"] for s in completion_statistics.values())))
    agg_row.append(str(sum(s["len_bin_4"] for s in completion_statistics.values())))
    agg_row.append(str(sum(s["len_bin_5"] for s in completion_statistics.values())))
    agg_row.append(str(sum(s["think_count"] for s in completion_statistics.values())))

    # Annotation stats aggregation
    for _, stats_dict in annotation_statistics:
        agg_row.append("")  # Empty cell for annotator name
        all_scores = [s for model_name in model_names if model_name in stats_dict for s in stats_dict[model_name].get("scores", [])]
        if all_scores:
            agg_row.append(f"{np.mean(all_scores):.2f}")
            agg_row.append(f"{np.std(all_scores):.2f}")
        else:
            agg_row.extend(["0.00", "0.00"])
        agg_row.extend([str(sum(stats_dict[model_name].get("num_best", 0) for model_name in model_names if model_name in stats_dict)),
                        str(sum(stats_dict[model_name].get("num_worst", 0) for model_name in model_names if model_name in stats_dict))])

    table_lines.append("| " + " | ".join(agg_row) + " |")

    return "\n".join(table_lines)

if __name__ == "__main__":
    load_dotenv(".env")
    load_dotenv(".env.local")
    huggingface_hub.login(os.getenv("HF_TOKEN"))

    if not os.path.exists("statistics.json"):
        # Calculate statistics for the datasets
        if not os.path.exists("/iopsstor/scratch/cscs/smarian/datasets/_analysis/skywork_llama_3.3_70b/first_sample.json"):
            skywork_llama_3_3_70b = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/5_merged_annotated_completions/skywork_with_small/llama_3.3_70b")
            skywork_llama_3_3_70b = get_lengths(skywork_llama_3_3_70b) 
            save_dataset(skywork_llama_3_3_70b, "/iopsstor/scratch/cscs/smarian/datasets/_analysis/skywork_llama_3.3_70b")
        else:
            skywork_llama_3_3_70b = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/_analysis/skywork_llama_3.3_70b")
        if not os.path.exists("/iopsstor/scratch/cscs/smarian/datasets/_analysis/skywork_qwen_32b/first_sample.json"):
            skywork_qwen_32b = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/5_merged_annotated_completions/skywork_with_small/qwen_3_32b")
            skywork_qwen_32b = get_lengths(skywork_qwen_32b)
            save_dataset(skywork_qwen_32b, "/iopsstor/scratch/cscs/smarian/datasets/_analysis/skywork_qwen_32b")
        else:
            skywork_qwen_32b = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/_analysis/skywork_qwen_32b")
        if not os.path.exists("/iopsstor/scratch/cscs/smarian/datasets/_analysis/skywork_qwen_235b/first_sample.json"):
            skywork_qwen_235b = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/5_merged_annotated_completions/skywork_with_small/qwen_3_235b")
            skywork_qwen_235b = get_lengths(skywork_qwen_235b)
            save_dataset(skywork_qwen_235b, "/iopsstor/scratch/cscs/smarian/datasets/_analysis/skywork_qwen_235b")
        else:
            skywork_qwen_235b = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/_analysis/skywork_qwen_235b")
        if not os.path.exists("/iopsstor/scratch/cscs/smarian/datasets/_analysis/skywork_skywork_rm/first_sample.json"):
            skywork_skywork_rm = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/5_merged_annotated_completions/skywork_with_small/qwen_3_8b_skywork_rm")
            skywork_skywork_rm = get_lengths(skywork_skywork_rm)
            save_dataset(skywork_skywork_rm, "/iopsstor/scratch/cscs/smarian/datasets/_analysis/skywork_skywork_rm")
        else:
            skywork_skywork_rm = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/_analysis/skywork_skywork_rm")
        if not os.path.exists("/iopsstor/scratch/cscs/smarian/datasets/_analysis/ultrafeedback_llama_3.3_70b/first_sample.json"):
            ultrafeedback_llama_3_3_70b = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/5_merged_annotated_completions/ultrafeedback_with_small/llama_3.3_70b")
            ultrafeedback_llama_3_3_70b = get_lengths(ultrafeedback_llama_3_3_70b)  
            save_dataset(ultrafeedback_llama_3_3_70b, "/iopsstor/scratch/cscs/smarian/datasets/_analysis/ultrafeedback_llama_3.3_70b")
        else:
            ultrafeedback_llama_3_3_70b = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/_analysis/ultrafeedback_llama_3.3_70b")
        if not os.path.exists("/iopsstor/scratch/cscs/smarian/datasets/_analysis/ultrafeedback_qwen_32b/first_sample.json"):
            ultrafeedback_qwen_32b = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/5_merged_annotated_completions/ultrafeedback_with_small/qwen_3_32b")
            ultrafeedback_qwen_32b = get_lengths(ultrafeedback_qwen_32b)  
            save_dataset(ultrafeedback_qwen_32b, "/iopsstor/scratch/cscs/smarian/datasets/_analysis/ultrafeedback_qwen_32b")
        else:
            ultrafeedback_qwen_32b = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/_analysis/ultrafeedback_qwen_32b")
        if not os.path.exists("/iopsstor/scratch/cscs/smarian/datasets/_analysis/ultrafeedback_qwen_235b/first_sample.json"):
            ultrafeedback_qwen_235b = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/5_merged_annotated_completions/ultrafeedback_with_small/qwen_3_235b")
            ultrafeedback_qwen_235b = get_lengths(ultrafeedback_qwen_235b)  
            save_dataset(ultrafeedback_qwen_235b, "/iopsstor/scratch/cscs/smarian/datasets/_analysis/ultrafeedback_qwen_235b")
        else:
            ultrafeedback_qwen_235b = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/_analysis/ultrafeedback_qwen_235b")
        if not os.path.exists("/iopsstor/scratch/cscs/smarian/datasets/_analysis/ultrafeedback_skywork_rm/first_sample.json"):
            ultrafeedback_skywork_rm = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/5_merged_annotated_completions/ultrafeedback_with_small/qwen_3_8b_skywork_rm")
            ultrafeedback_skywork_rm = get_lengths(ultrafeedback_skywork_rm)  
            save_dataset(ultrafeedback_skywork_rm, "/iopsstor/scratch/cscs/smarian/datasets/_analysis/ultrafeedback_skywork_rm")
        else:
            ultrafeedback_skywork_rm = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/_analysis/ultrafeedback_skywork_rm")

        # Calculate statistics for the datasets
        skywork_llama_3_3_70b_stats = calculate_statistics(skywork_llama_3_3_70b)
        skywork_qwen_32b_stats = calculate_statistics(skywork_qwen_32b)
        skywork_qwen_235b_stats = calculate_statistics(skywork_qwen_235b)
        skywork_skywork_rm_stats = calculate_statistics(skywork_skywork_rm)
        ultrafeedback_llama_3_3_70b_stats = calculate_statistics(ultrafeedback_llama_3_3_70b)
        ultrafeedback_qwen_32b_stats = calculate_statistics(ultrafeedback_qwen_32b)
        ultrafeedback_qwen_235b_stats = calculate_statistics(ultrafeedback_qwen_235b)
        ultrafeedback_skywork_rm_stats = calculate_statistics(ultrafeedback_skywork_rm)

        all_stats = {
            "skywork_llama_3.3_70b": skywork_llama_3_3_70b_stats,
            "skywork_qwen_32b": skywork_qwen_32b_stats,
            "skywork_qwen_235b": skywork_qwen_235b_stats,
            "skywork_skywork_rm": skywork_skywork_rm_stats,
            "ultrafeedback_llama_3.3_70b": ultrafeedback_llama_3_3_70b_stats,
            "ultrafeedback_qwen_32b": ultrafeedback_qwen_32b_stats,
            "ultrafeedback_qwen_235b": ultrafeedback_qwen_235b_stats,
            "ultrafeedback_skywork_rm": ultrafeedback_skywork_rm_stats
        }  

        with open("statistics.json", "w") as f:
            json.dump(all_stats, f, indent=4)
    else:
        with open("statistics.json", "r") as f:
            all_stats = json.load(f)

    # Generate and print the table for skywork
    completion_stats = all_stats["skywork_qwen_235b"]
    annotation_stats = [
        ("Qwen 3 235B", all_stats["skywork_qwen_235b"]),
        ("Skywork RM", all_stats["skywork_skywork_rm"]),
        ("Llama 3.3 70B", all_stats["skywork_llama_3.3_70b"]),
        ("Qwen 3 32B", all_stats["skywork_qwen_32b"]),
    ]
    skywork_table = generate_table(completion_stats, annotation_stats)

    # Generate and print the table for ultrafeedback
    completion_stats = all_stats["ultrafeedback_qwen_235b"]
    annotation_stats = [
        ("Qwen 3 235B", all_stats["ultrafeedback_qwen_235b"]),
        ("Skywork RM", all_stats["ultrafeedback_skywork_rm"]),
        ("Llama 3.3 70B", all_stats["ultrafeedback_llama_3.3_70b"]),
        ("Qwen 3 32B", all_stats["ultrafeedback_qwen_32b"]),
    ]
    ultrafeedback_table = generate_table(completion_stats, annotation_stats)

    with open("analysis.md", "w") as f:
        f.write("# Completions & Annotation Analysis\n\n")

        f.write("## Skywork\n\n")
        f.write(skywork_table)
        f.write("\n")

        f.write("## Ultrafeedback\n\n")
        f.write(ultrafeedback_table)
        f.write("\n")

    exit(0)

    # Count Number of Times Min and Max
    our_skywork_qwen_number_min_max = defaultdict(lambda: [0, 0])
    for i in range(len(our_skywork_qwen_scores["Qwen/Qwen3-235B-A22B"])):
        scores = [(model, our_skywork_qwen_scores[model][i]) for model in our_skywork_qwen_scores]
        min_model, min_score = min(scores, key=lambda x: x[1])
        max_model, max_score = max(scores, key=lambda x: x[1])
        our_skywork_qwen_number_min_max[min_model][0] += 1
        our_skywork_qwen_number_min_max[max_model][1] += 1

    our_skywork_skywork_rm_number_min_max = defaultdict(lambda: [0, 0])
    for i in range(len(our_skywork_skywork_rm_scores["Qwen/Qwen3-235B-A22B"])):
        scores = [(model, our_skywork_skywork_rm_scores[model][i]) for model in our_skywork_skywork_rm_scores]
        min_model, min_score = min(scores, key=lambda x: x[1])
        max_model, max_score = max(scores, key=lambda x: x[1])
        our_skywork_skywork_rm_number_min_max[min_model][0] += 1
        our_skywork_skywork_rm_number_min_max[max_model][1] += 1

    original_skywork_qwen_number_min_max = defaultdict(lambda: [0, 0])
    for i in range(len(original_skywork_qwen_scores["chosen"])):
        scores = [(model, original_skywork_qwen_scores[model][i]) for model in original_skywork_qwen_scores]
        min_model, min_score = min(scores, key=lambda x: x[1])
        max_model, max_score = max(scores, key=lambda x: x[1])
        original_skywork_qwen_number_min_max[min_model][0] += 1
        original_skywork_qwen_number_min_max[max_model][1] += 1

    original_skywork_skywork_rm_number_min_max = defaultdict(lambda: [0, 0])
    for i in range(len(original_skywork_skywork_rm_scores["chosen"])):
        scores = [(model, original_skywork_skywork_rm_scores[model][i]) for model in original_skywork_skywork_rm_scores]
        min_model, min_score = min(scores, key=lambda x: x[1])
        max_model, max_score = max(scores, key=lambda x: x[1])
        original_skywork_skywork_rm_number_min_max[min_model][0] += 1
        original_skywork_skywork_rm_number_min_max[max_model][1] += 1

    # Create Output File
    analysis_file = open("skywork_score_analysis.md", "w")

    analysis_file.write("# Our Completions - Qwen vs Skywork RM Annotations \n\n")
    analysis_file.write("| **Model** | **Qwen** | Average | Min | Max | #Best | #Worst | **Skywork** | Average | Min | Max | #Best | #Worst |\n")
    analysis_file.write("|-------|------|---------|-----|-----|-------|--------|---------|---------|-----|-----|-------|--------|\n")

    iterator = list(zip(our_skywork_qwen_scores.items(), our_skywork_skywork_rm_scores.items()))
    iterator.sort(key=lambda x: sum(x[0][1]) / len(x[0][1]), reverse=True)

    for (model_name, qwen_scores), (_, skywork_scores) in iterator:
        avg_score = f"{sum(qwen_scores)/len(qwen_scores):.3f}"
        min_score = f"{min(qwen_scores):.3f}"
        max_score = f"{max(qwen_scores):.3f}"
        skywork_avg = f"{sum(skywork_scores)/len(skywork_scores):.3f}"
        skywork_min = f"{min(skywork_scores):.3f}"
        skywork_max = f"{max(skywork_scores):.3f}"
        qwen_worst, qwen_best = our_skywork_qwen_number_min_max[model_name]
        skywork_worst, skywork_best = our_skywork_skywork_rm_number_min_max[model_name]
        analysis_file.write(f"| {model_name.split('/')[-1]} | | {avg_score} | {min_score} | {max_score} | {qwen_best} | {qwen_worst} | | {skywork_avg} | {skywork_min} | {skywork_max} | {skywork_best} | {skywork_worst} |\n")


    qwen_sum_scores = 0
    qwen_sum_lengths = 0
    qwen_min_score = 5
    qwen_max_score = -5
    for model_name, scores in our_skywork_qwen_scores.items():
        qwen_sum_scores += sum(scores)
        qwen_sum_lengths += len(scores)
        qwen_min_score = min(qwen_min_score, min(scores))
        qwen_max_score = max(qwen_max_score, max(scores))

    skywork_sum_scores = 0
    skywork_sum_lengths = 0
    skywork_min_score = 5
    skywork_max_score = -5
    for model_name, scores in our_skywork_skywork_rm_scores.items():
        skywork_sum_scores += sum(scores)
        skywork_sum_lengths += len(scores)
        skywork_min_score = min(skywork_min_score, min(scores))
        skywork_max_score = max(skywork_max_score, max(scores))

    analysis_file.write(f"| Aggregated | | {qwen_sum_scores / qwen_sum_lengths:.3f} | {qwen_min_score:.3f} | {qwen_max_score:.3f} | | | | {skywork_sum_scores / skywork_sum_lengths:.3f} | {skywork_min_score:.3f} | {skywork_max_score:.3f} | | |\n")
 

    analysis_file.write("# Original Completions - Qwen vs Skywork RM Annotations \n\n")
    analysis_file.write("| **Model** | **Qwen** | Average | Min | Max | #Best | #Worst | **Skywork** | Average | Min | Max | #Best | #Worst |\n")
    analysis_file.write("|-------|------|---------|-----|-----|-------|--------|---------|---------|-----|-----|-------|--------|\n")
    for (model_name, qwen_scores), (_, skywork_scores) in zip(original_skywork_qwen_scores.items(), original_skywork_skywork_rm_scores.items()):
        avg_score = f"{sum(qwen_scores)/len(qwen_scores):.3f}"
        min_score = f"{min(qwen_scores):.3f}"
        max_score = f"{max(qwen_scores):.3f}"
        skywork_avg = f"{sum(skywork_scores)/len(skywork_scores):.3f}"
        skywork_min = f"{min(skywork_scores):.3f}"
        skywork_max = f"{max(skywork_scores):.3f}"
        qwen_worst, qwen_best = original_skywork_qwen_number_min_max[model_name]
        skywork_worst, skywork_best = original_skywork_skywork_rm_number_min_max[model_name]
        analysis_file.write(f"| {model_name.split('/')[-1]} | | {avg_score} | {min_score} | {max_score} | {qwen_best} | {qwen_worst} | | {skywork_avg} | {skywork_min} | {skywork_max} | {skywork_best} | {skywork_worst} |\n")

    qwen_sum_scores = 0
    qwen_sum_lengths = 0
    qwen_min_score = 5
    qwen_max_score = -5
    for model_name, scores in original_skywork_qwen_scores.items():
        qwen_sum_scores += sum(scores)
        qwen_sum_lengths += len(scores)
        qwen_min_score = min(qwen_min_score, min(scores))
        qwen_max_score = max(qwen_max_score, max(scores))

        print(sorted(scores)[:50])
    skywork_sum_scores = 0
    skywork_sum_lengths = 0
    skywork_min_score = 5
    skywork_max_score = -5
    for model_name, scores in original_skywork_skywork_rm_scores.items():
        skywork_sum_scores += sum(scores)
        skywork_sum_lengths += len(scores)
        skywork_min_score = min(skywork_min_score, min(scores))
        skywork_max_score = max(skywork_max_score, max(scores))


    analysis_file.write(f"| Aggregated | | {qwen_sum_scores / qwen_sum_lengths:.3f} | {qwen_min_score:.3f} | {qwen_max_score:.3f} | | | | {skywork_sum_scores / skywork_sum_lengths:.3f} | {skywork_min_score:.3f} | {skywork_max_score:.3f} | | |\n")
 
    analysis_file.close()