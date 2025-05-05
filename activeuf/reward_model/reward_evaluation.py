import sys
import datasets
from datasets import load_dataset
import argparse
import os
import json
import pandas as pd


def eval_rewardbench(model, dataset, split, chat_template, batch_size, load_json=False, logging=False):
    # Prepare result file path
    result_dir = "results_rewardbench"
    result_file = os.path.join(result_dir, f"{model}.json")
    os.makedirs(result_dir, exist_ok=True)

    # Build command
    command = [
        "rewardbench",
        f"--model={model}",
        f"--dataset={dataset}",
        f"--split={split}",
        f"--chat_template={chat_template}",
        f"--batch_size={batch_size}",
        f"--output_dir={result_dir}/"
    ]
    if load_json:
        command.append("--load_json")

    # Run the command and handle logging
    try:
        if logging:
            log_file = os.path.join(result_dir, f"{model}_{dataset.split("/")[-1]}log.txt")
            with open(log_file, "w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=log, text=True)
        else:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ RewardBench failed for {model} on {dataset} (code {e.returncode})")
        return

    # Load result if available
    if os.path.exists(result_file):
        with open(result_file) as f:
            result = json.load(f)
        acc = result.get("accuracy", "N/A")
        prompts = result.get("num_prompts", "N/A")
        print(f"✅ {model} | {dataset} | Prompts: {prompts} | Acc: {acc:.3f}")
    else:
        print(f"⚠️  No result file found for {model}.{' Check ' + log_file if logging else ''}")

### Debugging

def debug_rewardbench(model, dataset, split, chat_template, batch_size, load_json=False):
    from rewardbench.rewardbench import main
    sys.argv = [
        "rewardbench",
        f"--model={model}",
        f"--dataset={dataset}",
        f"--split={split}",
        f"--chat_template={chat_template}",
        f"--batch_size={batch_size}"
    ]
    if load_json:
        sys.argv.append("--load_json")
    print(sys.argv)
    main()



def split_reward_bench_dataset(dataset_path="allenai/reward-bench", split="filtered", column="subset"):

    def convert_to_chat_format(example):
        # Wrap flat strings into chat-format lists
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        # Convert to chat-style turns
        user_msg = {"role": "user", "content": prompt}
        chosen_msg = {"role": "assistant", "content": chosen}
        rejected_msg = {"role": "assistant", "content": rejected}

        # Add fields in RewardBench-compatible format
        return {
            "prompt": prompt,
            "prompt_id": example.get("id", None),
            "chosen": [user_msg, chosen_msg],
            "rejected": [user_msg, rejected_msg],
            "messages": [user_msg, chosen_msg],
            "source": example.get("subset", example.get("source", "unknown")),
            "chosen_model": example.get("chosen_model", ""),
            "rejected_model": example.get("rejected_model", "")
        }

    # Load the dataset split
    dataset = load_dataset(dataset_path, split=split)

    # Get all unique sources
    sources = set(dataset[column])

    # Output directory
    os.makedirs(f"{dataset_path}-{split}", exist_ok=True)

    # Save each source as its own .jsonl file
    for src in sources:
        subset = dataset.filter(lambda x: x[column] == src)

        # Convert each example to chat format
        converted = [convert_to_chat_format(ex) for ex in subset]

        # Save as JSONL
        subset_path = os.path.join(f"{dataset_path}-{split}", f"{src}.json")
        with open(subset_path, "w") as f:
            for item in converted:
                f.write(json.dumps(item) + "\n")

        print(f"✅ Saved {len(converted)} examples to {subset_path}")



def split_ultrafeedback_dataset(dataset_path="allenai/ultrafeedback_binarized_cleaned", split="test_prefs", column="source"):
    # Load the dataset split
    dataset = load_dataset(dataset_path, split=split)

    # Get all unique sources
    sources = set(dataset[column])

    # Output directory
    os.makedirs(f"{dataset_path}-{split}", exist_ok=True)

    # Save each source as its own .jsonl file
    for src in sources:
        subset = dataset.filter(lambda x: x[column] == src)
        subset_path = os.path.join(f"{dataset_path}-{split}", f"{src}.json")
        subset.to_json(subset_path, orient="records", lines=True)
        print(f"Saved {len(subset)} examples to {subset_path}")



def evaluate_datasets(model, dataset_path, split, chat_template, batch_size, column="subset"):
     # Load the dataset split
    dataset = load_dataset(dataset_path, split=split)

    # Get all unique sources
    sources = set(dataset[column])
    for src in sources:
        local_dataset_path = os.path.join(f"{dataset_path}-{split}", f"{src}.json")
        eval_rewardbench(model, local_dataset_path, split, chat_template, batch_size, load_json=True)


def collect_results():
        # Load full score data
    df = pd.read_json("results/rewarded4_outputs.jsonl", lines=True)

    # Display example scores
    print(df[["prompt", "scores_chosen", "scores_rejected"]].head())

    # Load summary stats
    with open("results/rewarded4.json") as f:
        summary = json.load(f)

    print("Accuracy:", summary["accuracy"])
    print("Mean margin (approx):", 
        df["scores_chosen"].mean() - df["scores_rejected"].mean())


def main(config):

    dataset = config.dataset

    if dataset == "allenai/ultrafeedback_binarized_cleaned":
        split= "test_prefs"
        column = "source"
        split_ultrafeedback_dataset(dataset, split, column)
        #evaluate_datasets(config.model, dataset, split, config.chat_template, config.batch_size, column=column)
        #eval_rewardbench(config.model, dataset, split, config.chat_template, config.batch_size)

    elif dataset == "allenai/reward-bench":
        split= "filtered"
        column = "subset"
        split_reward_bench_dataset(dataset, split, column)
        #evaluate_datasets(config.model, dataset, split, config.chat_template, config.batch_size, column=column)
        #eval_rewardbench(config.model, dataset, split, config.chat_template, config.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train reward model using reward config YAML.")
    parser.add_argument("--model", type=str, required=True, help="Path to model config or default model like OpenAssistant/reward-model-deberta-v3-base.")
    parser.add_argument("--dataset", type=str,default ="allenai/reward-bench", help="Huggfingface dataset path on which rewardbench is evaluated.")
    parser.add_argument("--batch-size", type=int, default=3, help="Batch Size for reward evaluation.")
    parser.add_argument("--chat-template", type=str, default="raw", help="Chat template for reward evaluation.")
    parser.add_argument("--out-dir", type=str, default="reward-bench", help="Out dir for storing all dataset splits.")
    config = parser.parse_args()
    main(config)