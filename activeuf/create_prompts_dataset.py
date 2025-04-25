import argparse

from datasets import load_dataset

from activeuf.schemas import Prompt
from activeuf.utils import get_logger

"""
This script downlaods a dataset from HuggingFace and processes it into a dataset of prompts (that follows the Prompt schema in `schemas.py`).

Example run command from project root:
    python -m activeuf.create_prompts_dataset \
        --dataset_path allenai/ultrafeedback_binarized_cleaned \
        --dataset_split train_prefs \
        --output_path datasets/input_datasets/ultrafeedback_binarized_cleaned/train.jsonl
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="The HuggingFace path of the dataset to extract prompts from (e.g. allenai/ultrafeedback_binarized_cleaned)")
    parser.add_argument("--dataset_split", type=str, help="The split of the dataset to use (e.g. train_prefs, test_prefs)")
    parser.add_argument("--output_path", type=str, help="The path for saving the extracted prompts")
    return parser.parse_args()

def construct_prompt_from_sample(sample: dict) -> dict:
    """
    Construct a prompt from the information in the input sample. 
    You may need to modify this function if you're using a different dataset.
    """
    return Prompt(**sample).model_dump()

logger = get_logger(__name__)
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
    prompts.to_json(args.output_path, orient="records", lines=True)