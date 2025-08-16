import argparse
import glob
import json
import os.path as path

from datasets import load_from_disk

from activeuf.schemas import PromptWithCompletions
from activeuf.configs import *
from activeuf.utils import *

logger = get_logger(__name__)

"""
This script takes a folder of datasets with completions as input and merges them into a single dataset.

Example run command:
    python -m activeuf.merge_completions \
        --datasets_path datasets/completions \
        --output_path datasets/merged_completions
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_path", type=str, required=True, help="The path to the folder of datasets with completions")
    parser.add_argument("--output_path", type=str, help="Where to save the merged dataset")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f"{args.datasets_path.rstrip('/')}-merged"

    assert not path.exists(args.output_path), f"Output path {args.output_path} already exists"

    return args

def extend_completions(sample: dict) -> dict:
    # determine which models have already generated completions for this sample
    models_done = {_["model"] for _ in sample["completions"]}
    
    # identify new completions by models not in models_done
    new_completions = [
        _ for _ in sample["new_completions"]
        if _["model"] not in models_done
    ]
    sample["completions"] += new_completions
    return sample

if __name__ == "__main__":
    args = parse_args()

    # get paths to all datasets in specified folder
    dataset_paths = glob.glob(path.join(args.datasets_path, "*"))
    logger.info(f"Number of datasets found: {len(dataset_paths)}")

    # iteratively merge the datasets
    merged = None
    for i, dataset_path in enumerate(dataset_paths):
        logger.info(f"Loading and merging dataset #{i}: {dataset_path}")
        dataset = load_from_disk(dataset_path)

        # sanity check: dataset complies with PromptWithCompletions schema
        for sample in dataset:
            PromptWithCompletions.model_validate(sample)

        # init merged dataset
        if merged is None:
            merged = dataset
            continue

        # get completions to be added to each sample in merged
        prompt_id2dataset_idx = {
            prompt_id: i
            for i, prompt_id in enumerate(dataset["prompt_id"])
        }
        new_completions = [
            dataset[prompt_id2dataset_idx[prompt_id]]["completions"]
            for prompt_id in merged["prompt_id"]
        ]

        # add completions by models not in models_done to merged
        merged = merged.add_column("new_completions", new_completions)
        merged = merged.map(extend_completions)
        merged = merged.remove_columns("new_completions")

    # export merged dataset
    logger.info(f"Saving merged dataset to {args.output_path}")
    merged.save_to_disk(args.output_path)

    # Export args
    args_path = path.join(args.output_path, "args.json")
    with open(args_path, "w") as f_out:
        json.dump(vars(args), f_out)