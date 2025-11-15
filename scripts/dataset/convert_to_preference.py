import argparse
import os
import random
import json

from datasets import load_from_disk, Dataset

from activeuf.utils import set_seed, get_logger
from activeuf.configs import SEED

logger = get_logger(__name__)

"""
This script can be used to convert a dataset with completions and annotations for every completion to turn it into a binarized dataset.
For this the script uses:
    - Randomly sampling two completions and choosing the one with the highest overall score as chosen and the other as rejected.
    - The approach described in the ultrafeedback paper (https://arxiv.org/abs/2310.01377) which randomly samples 4 models/completions and chooses the best as chosen and randomly samples the rejected from the remaining 3.

Example run command:
    python -m activeuf.convert_to_preference \
        --input_path /iopsstor/scratch/cscs/smarian/datasets/ultrafeedback_annotated

python -m activeuf.convert_to_preference --input_path /iopsstor/scratch/cscs/smarian/datasets/ultrafeedback_annotated
"""


def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the annotated dataset."
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to save the ultrafeedback-style dataset."
    )
    parser.add_argument(
        "--formatted_output_path",
        type=str,
        help="Path to save the preference formatted dataset.",
    )

    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    if not args.output_path:
        args.output_path = args.input_path
    assert not os.path.exists(f"{args.output_path}_random"), (
        f"Output path {args.output_path}_random already exists"
    )
    assert not os.path.exists(f"{args.output_path}_ultrafeedback"), (
        f"Output path {args.output_path}_ultrafeedback already exists"
    )
    assert not os.path.exists(f"{args.output_path}_max_min"), (
        f"Output path {args.output_path}_max_min already exists"
    )

    return args


def convert_to_ultrafeedback(sample):
    """
    Converts a sample from a dataset containing completions for every completion into a binarized datastyle
    using the approach described in the ultrafeedback paper (https://arxiv.org/abs/2310.01377).
    """

    num_completions = len(sample["completions"])
    if num_completions < 4:
        raise ValueError(
            "Need at least 4 completions to convert to ultrafeedback style"
        )

    # Randomly sample 4 completions and sort the by overall score (descending)
    sampled_indices = random.sample(range(num_completions), 4)
    sampled_completions = [sample["completions"][i] for i in sampled_indices]
    sorted_completions = sorted(
        sampled_completions, key=lambda x: x["overall_score"], reverse=True
    )

    # Choose the best completion as chosen
    chosen_completion = sorted_completions[0]
    rejected_completions = random.choice(sorted_completions[1:])

    return {
        "prompt": sample["prompt"],
        "prompt_id": sample["prompt_id"],
        "rejected": rejected_completions["response_text"],
        "rejected_model": rejected_completions["model"],
        "rejected_score": rejected_completions["overall_score"],
        "chosen": chosen_completion["response_text"],
        "chosen_model": chosen_completion["model"],
        "chosen_score": chosen_completion["overall_score"],
    }


def convert_to_random(sample):
    """
    Converts a sample from a dataset containing completions for every completion into a binarized datastyle
    by randomly sampling 2 completions and choosing one as chosen and the other as rejected.
    """

    num_completions = len(sample["completions"])
    if num_completions < 2:
        raise ValueError("Need at least 2 completions to convert to random style")

    # Randomly sample 2 completions
    sampled_indices = random.sample(range(num_completions), 2)
    sampled_completions = [sample["completions"][i] for i in sampled_indices]
    sampled_completions = sorted(
        sampled_completions, key=lambda x: x["overall_score"], reverse=True
    )

    # Choose one completion as chosen and the other as rejected
    chosen_completion = sampled_completions[0]
    rejected_completion = sampled_completions[1]

    return {
        "prompt": sample["prompt"],
        "prompt_id": sample["prompt_id"],
        "chosen": chosen_completion["response_text"],
        "chosen_model": chosen_completion["model"],
        "chosen_score": chosen_completion["overall_score"],
        "rejected": rejected_completion["response_text"],
        "rejected_model": rejected_completion["model"],
        "rejected_score": rejected_completion["overall_score"],
    }


def convert_to_max_min(sample):
    """
    Converts a sample from a dataset containing completions for every completion into a binarized datastyle
    by randomly sampling 2 completions and choosing one as chosen and the other as rejected.
    """

    num_completions = len(sample["completions"])
    if num_completions < 2:
        raise ValueError("Need at least 2 completions to convert to random style")

    # Randomly sample 2 completions
    completions = sample["completions"]
    completions = sorted(completions, key=lambda x: x["overall_score"], reverse=True)

    # Choose one completion as chosen and the other as rejected
    chosen_completion = completions[0]
    rejected_completion = completions[-1]

    return {
        "prompt": sample["prompt"],
        "prompt_id": sample["prompt_id"],
        "chosen": chosen_completion["response_text"],
        "chosen_model": chosen_completion["model"],
        "chosen_score": chosen_completion["overall_score"],
        "rejected": rejected_completion["response_text"],
        "rejected_model": rejected_completion["model"],
        "rejected_score": rejected_completion["overall_score"],
    }


def first_sample_of(dataset_path):
    dataset = load_from_disk(dataset_path)
    with open(os.path.join(dataset_path, "first_sample.json"), "w") as f:
        json.dump(dataset[0], f, indent=2)


def to_preference_format(dataset_path, output_path):
    dataset = load_from_disk(dataset_path)

    def to_conversation_format(example):
        return {
            "chosen": [
                {
                    "role": "user",
                    "content": example["prompt"],
                },
                {
                    "role": "assistant",
                    "content": example["chosen"],
                },
            ],
            "rejected": [
                {
                    "role": "user",
                    "content": example["prompt"],
                },
                {
                    "role": "assistant",
                    "content": example["rejected"],
                },
            ],
        }

    dataset = dataset.map(to_conversation_format, remove_columns=["chosen", "rejected"])
    dataset.save_to_disk(output_path)
    first_sample_of(output_path)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.seed:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    logger.info("Loading dataset")
    dataset = load_from_disk(args.input_path)

    logger.info("Converting dataset to random style")
    random_dataset = dataset.map(
        convert_to_random,
        remove_columns=dataset.column_names,
        desc="Converting to random style",
        load_from_cache_file=False,
    )

    logger.info("Converting dataset to ultrafeedback style")
    ultrafeedback_dataset = dataset.map(
        convert_to_ultrafeedback,
        remove_columns=dataset.column_names,
        desc="Converting to ultrafeedback style",
        load_from_cache_file=False,
    )

    logger.info("Converting dataset to max-min style")
    max_min_dataset = dataset.map(
        convert_to_max_min,
        remove_columns=dataset.column_names,
        desc="Converting to max-min style",
        load_from_cache_file=False,
    )

    print(len(random_dataset))
    print(len(ultrafeedback_dataset))
    print(len(max_min_dataset))

    logger.info("Saving datasets")
    random_dataset.save_to_disk(f"{args.output_path}/random")
    first_sample_of(f"{args.output_path}/random")
    ultrafeedback_dataset.save_to_disk(f"{args.output_path}/ultrafeedback")
    first_sample_of(f"{args.output_path}/ultrafeedback")
    max_min_dataset.save_to_disk(f"{args.output_path}/max_min")
    first_sample_of(f"{args.output_path}/max_min")

    if args.formatted_output_path:
        logger.info("Converting datasets to preference format")
        to_preference_format(
            f"{args.output_path}/random", f"{args.formatted_output_path}/random"
        )
        to_preference_format(
            f"{args.output_path}/ultrafeedback",
            f"{args.formatted_output_path}/ultrafeedback",
        )
        to_preference_format(
            f"{args.output_path}/max_min", f"{args.formatted_output_path}/max_min"
        )
