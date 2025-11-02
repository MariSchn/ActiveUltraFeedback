import os
import argparse
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import json


def calculate_overall_score(annotation):
    score = 0.0
    for aspect, output in annotation.items():
        for score_value, weight in output.items():
            if weight:
                score += float(score_value) * float(weight)
    return score / len(annotation)


def combine_annotations(annotations_folder, completions_folder, output_folder):
    datasets_annotation = []
    datasets_completion = []
    foldernames = []
    for foldername in tqdm(sorted(os.listdir(annotations_folder))):
        dataset = load_from_disk(os.path.join(annotations_folder, foldername))
        print(f"Loaded annotation dataset from {foldername} with {len(dataset)} entries")
        datasets_annotation.append(dataset)
        foldernames.append(foldername)

    for i, foldername in enumerate(tqdm(sorted(os.listdir(completions_folder)))):
        assert foldername == foldernames[i], f"Folder ordering does not match got {foldername} expected {foldernames[i]}"
        dataset = load_from_disk(os.path.join(completions_folder, foldername))
        datasets_completion.append(dataset)

    completions_len = len(datasets_completion[0])
    for foldername, dataset in zip(foldernames, datasets_annotation):
        if len(dataset) == completions_len:
            print(f"\033[92mLoaded annotation dataset from {foldername} with {len(dataset)} entries\033[0m")
        else:
            print(f"\033[91mLoaded annotation dataset from {foldername} with {len(dataset)} entries (expected {completions_len})\033[0m")

    assert len(datasets_annotation) == len(
        datasets_completion), "Number of annotation datasets must match number of completion datasets"

    combined_dataset = []
    for i in tqdm(range(len(datasets_annotation[0]))):
        new_row = {
            "prompt": datasets_completion[0][i]["prompt"],
            "prompt_id": datasets_completion[0][i]["prompt_id"],
            "source": datasets_completion[0][i]["source"],
            "completions": []
        }

        for j in range(len(datasets_annotation)):
            dataset = datasets_completion[j]

            if j < len(datasets_annotation) - 1:
                assert dataset[i]["prompt_id"] == datasets_completion[j +
                                                                      1][i]["prompt_id"], "Prompt ID ordering does not match across datasets"

            assert datasets_annotation[j][i]["prompt_id"] == dataset[i][
                "prompt_id"], "Prompt ID ordering does not match across annotation and completion datasets"

            completion = dataset[i]["completions"][0]
            assert len(dataset[i]["completions"]
                       ) == 1, "Expected exactly one completion per prompt"

            try: 
                annotations = datasets_annotation[j][i]["annotation"]
            except Exception as e:
                # print(f"Error accessing annotation for dataset {j}, index {i}: {e}")
                annotations = []
            
            try:
                overall_score = calculate_overall_score(annotations)
            except Exception as e:
                # print(f"Error calculatincg overall score for dataset {j}, index {i}: {e}")
                overall_score = datasets_annotation[j][i]["completions"][0]["overall_score"]
            
            new_row["completions"].append({
                "annotations": annotations,
                "critique": "",  # not required for our purposes
                "messages": completion["messages"],
                "model": completion["model"],
                "overall_score": overall_score,
                "principle": completion["principle"],
                "response_text": completion["response_text"],
                "system_prompt": completion["system_prompt"],
            })
        combined_dataset.append(new_row)

    combined_dataset = Dataset.from_list(combined_dataset)
    if output_folder:
        combined_dataset.save_to_disk(output_folder)
        print(f"Combined dataset saved to {output_folder}")

    # Save the first sample of the combined dataset to a JSON file
    first_sample_path = os.path.join(output_folder, "first_sample.json")
    with open(first_sample_path, "w") as f:
        json.dump(combined_dataset[0], f, indent=2)
    print(f"First sample saved to {first_sample_path}")

    return combined_dataset


def main():
    parser = argparse.ArgumentParser(description="Combine annotated completions datasets.")
    parser.add_argument("--annotations_folder", type=str, required=True, help="Path to the folder containing annotation datasets.")
    parser.add_argument("--completions_folder", type=str, required=True, help="Path to the folder containing completion datasets.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save the combined dataset.")

    args = parser.parse_args()
    annotations_folder = args.annotations_folder
    completions_folder = args.completions_folder
    output_folder = args.output_folder

    combined_dataset = combine_annotations(
        annotations_folder, completions_folder, output_folder)
    print(combined_dataset)
    print(combined_dataset.features)

    # Save the first sample of the combined dataset to a JSON file
    first_sample_path = os.path.join(output_folder, "first_sample.json")
    with open(first_sample_path, "w") as f:
        json.dump(combined_dataset[0], f, indent=2)
    print(f"First sample saved to {first_sample_path}")

    # checker_folder = "/iopsstor/scratch/cscs/dmelikidze/datasets/ultrafeedback_annotated_combined_new_qwen10/Qwen3-32B"
    # dataset = load_from_disk(output_folder)
    # print(dataset)
    # dataset2 = load_from_disk(checker_folder)
    # print(dataset2)
    # for j in range(17):
    #     if dataset[6464]["completions"][j]["model"] == "Qwen/Qwen3-32B":
    #         print(dataset[6464]["completions"][j]["annotations"],
    #               dataset[6464]["completions"][j]["model"])
    # print("---")
    # print(dataset2[6464]["annotation"], dataset2[6464]["model"])

    # dataset = load_from_disk(
    #     "/iopsstor/scratch/cscs/dmelikidze/datasets/checking/")
    # print(dataset)
    # print(dataset.features)
    # print(dataset[0]["completions"][0].keys())
    # dataset2 = load_from_disk(
    #     "/iopsstor/scratch/cscs/dmelikidze/datasets/completions_combined/")
    # print(dataset2)
    # print(dataset2[0]["completions"][0].keys())

    # dataset3 = load_from_disk(
    #     "/iopsstor/scratch/cscs/dmelikidze/datasets/completions_all/c4ai-command-a-03-2025")
    # print(dataset3)
    # print(dataset3[0]["completions"][0].keys())

    # dataset4 = load_from_disk(
    #     "/iopsstor/scratch/cscs/dmelikidze/datasets/ultrafeedback_annotated_combined_new_qwen10/c4ai-command-a-03-2025")

    # print(dataset4)
    # print(dataset4[0]["annotation"])


if __name__ == "__main__":
    main()