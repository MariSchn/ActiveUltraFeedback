import subprocess
import sys
import datasets
from datasets import load_dataset
import argparse
import os
import json
import pandas as pd


def eval_rewardbench(model, dataset, split, chat_template, batch_size, load_json=False):
    # Prepare result file path
    result_dir = "results"
    result_file = os.path.join(result_dir, f"{model}.json")

    # Optional log file
    log_file = os.path.join(result_dir, f"{model}_log.txt")
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

    # Run the command and capture stdout/stderr
    with open(log_file, "w") as log:
        try:
            subprocess.run(command, check=True, stdout=log, stderr=log, text=True)
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
        print(f"⚠️  No result file found for {model}. Check {log_file}")

### Debugging

def debug_rewardbench(model, dataset, split, chat_template, batch_size):
    from rewardbench.rewardbench import main
    sys.argv = [
        "rewardbench",
        f"--model={model}",
        f"--dataset={dataset}",
        f"--split={split}",
        f"--chat_template={chat_template}",
        f"--batch_size={batch_size}"
    ]
    print(sys.argv)
    main()


def split_dataset(split, output_dir="ultrafeedback"):
    # Load the dataset split (choose 'train_sft', 'test_sft', etc.)
    dataset = load_dataset("allenai/ultrafeedback_binarized_cleaned", split=split)

    # Get all unique sources
    sources = set(dataset["source"])

    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save each source as its own .jsonl file
    for src in sources:
        subset = dataset.filter(lambda x: x["source"] == src)
        subset_path = os.path.join(f"{output_dir}_{split}", f"{src}.json")
        subset.to_json(subset_path, orient="records", lines=True)
        print(f"Saved {len(subset)} examples to {subset_path}")


def evaluate_ultrafeedback_datasets(model, dataset, split, chat_template, batch_size, output_dir="ultrafeedback"):
     # Load the dataset split (choose 'train_sft', 'test_sft', etc.)
    dataset = load_dataset(dataset, split=split)

    # Get all unique sources
    sources = set(dataset["source"])
    for src in sources:
        local_dataset_path = os.path.join(f"{output_dir}_{split}", f"{src}.json")
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


#reward4


def main(config):
    split_dataset(config.split)
    #eval_rewardbench(config.model, config.dataset, config.split, config.chat_template, config.batch_size)
    splits = ["test_pref", "test_gen", "test_sft"]
    evaluate_ultrafeedback_datasets(config.model, config.dataset, config.split, config.chat_template, config.batch_size)
    eval_rewardbench(config.model, config.dataset, config.split, config.chat_template, config.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train reward model using reward config YAML.")
    parser.add_argument("--model", required=True, help="Path to model config or default model like OpenAssistant/reward-model-deberta-v3-base.")
    parser.add_argument("--dataset", default ="allenai/ultrafeedback_binarized_cleaned", help="Datset rewardbench to be evaluated on.")
    parser.add_argument("--split", default ="test_prefs", help="Split of rewardbench dataset.")
    parser.add_argument("--batch-size", default=3, help="Batch Size for reward evaluation.")
    parser.add_argument("--chat-template", default="raw", help="Chat template for reward evaluation.")
    config = parser.parse_args()
    main(config)