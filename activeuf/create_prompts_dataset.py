import argparse
import os

from datasets import load_dataset

from activeuf.schemas import Prompt
from activeuf.utils import get_logger

logger = get_logger(__name__)

"""
This script downloads a dataset from HuggingFace and processes it into a dataset of prompts (that follows the Prompt schema in `schemas.py`).

Example run command from project root:
    python -m activeuf.create_prompts_dataset \
        --dataset_path allenai/ultrafeedback_binarized_cleaned \
        --dataset_split train_prefs
        --output_path 
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, help="The HuggingFace path of the dataset to extract prompts from (e.g. allenai/ultrafeedback_binarized_cleaned)")
    parser.add_argument("--dataset_split", type=str, help="The split of the dataset to use (e.g. train_prefs, test_prefs)")

    parser.add_argument("--output_path", type=str, help="Where to save the prompts dataset")

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = os.path.join("datasets", args.dataset_path)
        if args.dataset_split:
            args.output_path = os.path.join(args.output_path, args.dataset_split)
        logger.info(f"Exporting to {args.output_path} because no output path was provided")

    return args

def construct_prompt_from_sample(sample: dict) -> dict:
    """
    Construct a prompt from the information in the input sample. 
    You may need to modify this function if you're using a different dataset.
    """
    return Prompt(**sample).model_dump()

if __name__ == "__main__":
    args = parse_args()

    # Load HF data
    logger.info(f"Loading dataset from {args.dataset_path} (split={args.dataset_split})")
    dataset = load_dataset(args.dataset_path, split=args.dataset_split)

    # Apply Prompt schema to dataset and export
    logger.info(f"Processing dataset into prompts")
    prompts = dataset.map(construct_prompt_from_sample, remove_columns=dataset.column_names)

    # Export prompts dataset
    logger.info(f"Saving prompts to {args.output_path}")
    prompts.save_to_disk(args.output_path)