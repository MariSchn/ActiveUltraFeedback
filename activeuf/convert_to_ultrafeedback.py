import argparse
import os
import random

from datasets import load_from_disk

from activeuf.utils import set_seed, get_logger
from activeuf.configs import SEED

logger = get_logger(__name__)

"""
This script can be used to convert a dataset with completions and annotations for every completion to turn it into a binarized dataset.
For this the script uses the approach described in the ultrafeedback paper (https://arxiv.org/abs/2310.01377), 
which randomly samples 4 models/completions and chooses the best as chosen and randomly samples the rejected from the remaining 3.

Example run command:
    python -m activeuf.convert_to_ultrafeedback \
        --input_path /iopsstor/scratch/cscs/smarian/datasets/allenai/ultrafeedback_binarized_cleaned/train_prefs-with-completions-sanitized-annotated-first \
        --output_path /iopsstor/scratch/cscs/smarian/datasets/allenai/ultrafeedback_binarized_cleaned/train_prefs-with-completions-sanitized-annotated-first-ultrafeedback

python -m activeuf.convert_to_ultrafeedback --input_path /iopsstor/scratch/cscs/smarian/datasets/allenai/ultrafeedback_binarized_cleaned/train_prefs-with-completions-sanitized-annotated-first --output_path /iopsstor/scratch/cscs/smarian/datasets/allenai/ultrafeedback_binarized_cleaned/train_prefs-with-completions-sanitized-annotated-first-ultrafeedback
"""


def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True, help="Path to the annotated dataset.")
    parser.add_argument("--output_path", type=str, help="Path to save the ultrafeedback-style dataset.")

    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")

    args = parser.parse_args()

    if not args.output_path:
        args.output_path = f"{args.input_path.rstrip('/')}-ultrafeedback"
    assert not os.path.exists(args.output_path), f"Output path {args.output_path} already exists"

    return args

def convert_to_ultrafeedback(sample):
    """
    Converts a sample from a dataset containing completions for every completion into a binarized datastyle
    using the approach described in the ultrafeedback paper (https://arxiv.org/abs/2310.01377).
    """

    num_completions = len(sample["completions"])
    if num_completions < 4:
        raise ValueError("Need at least 4 completions to convert to ultrafeedback style")
    
    # Randomly sample 4 completions and sort the by overall score (descending)
    sampled_indices = random.sample(range(num_completions), 4)
    sampled_completions = [sample["completions"][i] for i in sampled_indices]
    sorted_completions = sorted(sampled_completions, key=lambda x: x["overall_score"], reverse=True)

    # Choose the best completion as chosen
    chosen_completion = sorted_completions[0]
    rejected_completions = random.choice(sorted_completions[1:])

    return {
        "prompt": sample["prompt"],
        "prompt_id": sample["prompt_id"],
        "chosen": chosen_completion["response_text"],
        "chosen_model": chosen_completion["model"],
        "rejected": rejected_completions["response_text"],
        "rejected_model": rejected_completions["model"],
    }


if __name__ == "__main__":
    args = parse_args()

    if args.seed:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)
    
    logger.info("Loading dataset")
    dataset = load_from_disk(args.input_path)

    logger.info("Converting dataset to ultrafeedback style")
    ultrafeedback_dataset = dataset.map(
        convert_to_ultrafeedback, 
        remove_columns=dataset.column_names,
        desc="Converting to ultrafeedback style"
    )

    logger.info("Saving dataset")
    ultrafeedback_dataset.save_to_disk(args.output_path)
