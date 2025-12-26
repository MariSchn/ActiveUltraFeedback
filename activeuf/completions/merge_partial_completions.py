import argparse
import glob
import json
import os.path as path
import re
from collections import defaultdict

from datasets import concatenate_datasets, load_from_disk

from activeuf.schemas import PromptWithCompletions
from activeuf.utils import get_logger

logger = get_logger(__name__)

"""
This script takes a folder of partial (chunked) datasets with completions as input
and merges them into complete datasets, one per model.

The partial datasets are expected to follow the naming convention:
    {MODEL_NAME}_{CHUNK_INDEX}

For example:
    Qwen3-235B-A22B_0
    Qwen3-235B-A22B_1
    ...

Example run command:
    python -m activeuf.completions.merge_partial_completions \
        --datasets_path /iopsstor/scratch/cscs/smarian/datasets/1_partial_completions/tulu_3 \
        --output_path /iopsstor/scratch/cscs/smarian/datasets/2_full_completions/tulu_3
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets_path",
        type=str,
        required=True,
        help="The path to the folder of partial datasets with completions",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Where to save the merged datasets (one subfolder per model)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate each sample against the PromptWithCompletions schema",
    )
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f"{args.datasets_path.rstrip('/')}-merged"

    return args


def extract_model_name_and_chunk(folder_name: str) -> tuple[str, int]:
    """
    Extract model name and chunk index from folder name.
    Expected format: {MODEL_NAME}_{CHUNK_INDEX}
    """
    # Match the last underscore followed by digits as the chunk index
    match = re.match(r"^(.+)_(\d+)$", folder_name)
    if match:
        return match.group(1), int(match.group(2))
    else:
        # If no chunk index found, treat as a single non-chunked dataset
        return folder_name, -1


def group_datasets_by_model(datasets_path: str) -> dict[str, list[tuple[str, int]]]:
    """
    Group dataset paths by model name.
    Returns a dict mapping model_name -> list of (dataset_path, chunk_index)
    """
    dataset_paths = glob.glob(path.join(datasets_path, "*"))

    model_to_chunks = defaultdict(list)
    for dataset_path in dataset_paths:
        if not path.isdir(dataset_path):
            continue
        folder_name = path.basename(dataset_path)
        model_name, chunk_index = extract_model_name_and_chunk(folder_name)
        model_to_chunks[model_name].append((dataset_path, chunk_index))

    # Sort chunks by index for each model
    for model_name in model_to_chunks:
        model_to_chunks[model_name].sort(key=lambda x: x[1])

    return dict(model_to_chunks)


if __name__ == "__main__":
    args = parse_args()

    # Group datasets by model name
    model_to_chunks = group_datasets_by_model(args.datasets_path)

    logger.info(f"Found {len(model_to_chunks)} models with partial datasets:")
    for model_name, chunks in model_to_chunks.items():
        chunk_indices = [idx for _, idx in chunks]
        logger.info(
            f"  - {model_name}: {len(chunks)} chunks (indices: {chunk_indices})"
        )

    # Create output directory
    if not path.exists(args.output_path):
        import os

        os.makedirs(args.output_path, exist_ok=True)

    # Merge datasets for each model
    for model_name, chunks in model_to_chunks.items():
        logger.info(f"\nProcessing model: {model_name}")

        model_output_path = path.join(args.output_path, model_name)
        if path.exists(model_output_path):
            logger.warning(
                f"  Output path {model_output_path} already exists, skipping..."
            )
            continue

        # Load and concatenate all chunks
        datasets = []
        for dataset_path, chunk_index in chunks:
            logger.info(f"  Loading chunk {chunk_index}: {dataset_path}")
            dataset = load_from_disk(dataset_path)
            datasets.append(dataset)

        if len(datasets) == 0:
            logger.warning(f"  No datasets found for {model_name}, skipping...")
            continue

        # Concatenate all chunks
        logger.info(f"  Concatenating {len(datasets)} chunks...")
        merged = concatenate_datasets(datasets)

        # Validate if requested
        if args.validate:
            logger.info(f"  Validating {len(merged)} samples...")
            for sample in merged:
                PromptWithCompletions.model_validate(sample)

        # Sort by prompt_id for consistency
        logger.info(f"  Sorting by prompt_id...")
        merged = merged.sort("prompt_id")

        # Export merged dataset
        logger.info(
            f"  Saving merged dataset ({len(merged)} samples) to {model_output_path}"
        )
        merged.save_to_disk(model_output_path)

        # Export first sample
        first_sample_path = path.join(model_output_path, "first_sample.json")
        with open(first_sample_path, "w") as f_out:
            json.dump(merged[0], f_out, indent=2)

        # Export metadata
        metadata = {
            "model_name": model_name,
            "num_samples": len(merged),
            "num_chunks_merged": len(chunks),
            "chunk_indices": [idx for _, idx in chunks],
            "source_paths": [p for p, _ in chunks],
            "args": vars(args),
        }
        metadata_path = path.join(model_output_path, "merge_metadata.json")
        with open(metadata_path, "w") as f_out:
            json.dump(metadata, f_out, indent=2)

        logger.info(f"  Done! Merged {len(chunks)} chunks into {len(merged)} samples")

    logger.info("\n=== All models processed ===")
