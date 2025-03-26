import argparse
import random
import json
import os

from datasets import Dataset, load_dataset

"""
This script can be used to download a dataset from Huggingface and process it to be used by the UltraFeedback scripts.
The output of this script will be a json dataset WITHOUT LLM completions, these will be added by the main script (`main_vllm.py`).
This script adds the necessary fields "models" and "completions" and randomly samples `--num_models` models to be used for the completions.

To run it, simply pass the name of the dataset you want to download and process as an argument to the script
and optionally how many completions should be generated for each prompt (default = 4).
"""

# TODO: Add latest models
model_pool = [
    "gpt-4", "gpt-3.5-turbo", "bard", 
    "ultralm-65b", "wizardlm-30b", "vicuna-33b", "llama-2-70b-chat", 
    "ultralm-13b", "wizardlm-13b", "llama-2-13b-chat", 
    "wizardlm-7b", "alpaca-7b", "llama-2-7b-chat", 
    "falcon-40b-instruct", "starchat", "mpt-30b-chat", "pythia-12b",
    "gpt-2"  # ! Only used for testing, as it is a relatively small model that can be easily loaded. Remove for actual runs.
]

#  TODO: Add more datasets
# Maps from a dataset name to the corresponding path or Huggingface dataset
dataset_map = {
    "truthful_qa": {"path": "truthfulqa/truthful_qa", "name": "generation"},
}

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to download and process (e.g. truthful_qa)")
    parser.add_argument("--num_models", type=int, default=4, help="The number of models to use for completions for each sample")
    parser.add_argument("--seed", type=int, default=123, help="Seed for random sampling")
    args = parser.parse_args()

    # Set random seed
    if isinstance(args.seed, int):
        random.seed(args.seed)

    # Load dataset
    dataset = load_dataset(**dataset_map[args.dataset_name])

    # Extract the prompt from the dataset, as described in: https://github.com/OpenBMB/UltraFeedback/issues/6
    # TODO: Check if there is a better way to handle this for different datasets. Probably not though.
    if args.dataset_name == "truthful_qa":
        dataset = Dataset.from_dict({
            "instruction": dataset['validation']["question"],
            "correct_answers": dataset['validation']["correct_answers"],
            "incorrect_answers": dataset['validation']["incorrect_answers"],
        })

    # Add models to be used for completions for each sample and the "completions" filled which is to be filled by the main script
    dataset = dataset.map(
        lambda x: {
            "models": random.sample(model_pool, args.num_models), 
            "completions": [], 
        }, 
        desc = args.dataset_name,
    )

    # Save dataset
    os.makedirs("./completion_data", exist_ok=True)
    with open(f"./completion_data/{args.dataset_name}.json", "w") as f:
            json.dump([{k: v for k, v in data.items()} for data in dataset], f, indent=4)