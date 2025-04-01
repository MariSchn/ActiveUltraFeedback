import argparse
import os
import random
from tqdm import tqdm
from typing import Generator

from datasets import load_dataset

from activeuf.schemas import Sample
from activeuf.configs import DATASET_MAP, MODEL_POOL
from activeuf.utils import set_seed

# TODO: update this docstring
"""
This script downloads a dataset from HuggingFace and processes it to follow the input data schema (defined in `schemas.py`) for the ActiveUltraFeedback pipeline.
The output of this script is a jsonl dataset without LLM completions (which will be added by the next script `generate_completions.py`).

Example run command from project root:
    python -m activeuf.create_input_dataset --dataset_name truthful_qa
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to download and process (e.g. truthful_qa)")
    
    parser.add_argument("--num_models", type=int, default=4, help="The number of models to use for completions for each sample")
    parser.add_argument("--seed", type=int, default=123, help="Seed for random sampling")

    parser.add_argument("--output_dir", type=str, default="datasets/input_datasets/", help="The directory for exporting the input dataset")
    return parser.parse_args()

def prepare_input_data_iter(dataset_name: str) -> Generator[Sample, None, None]:
    # Given a supported dataset name, load the dataset from HF and extract the relevant fields
    
    # Get HF path for this dataset
    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets are: {list(DATASET_MAP.keys())}")
    hf_path = DATASET_MAP[dataset_name]

    # Load dataset from HF
    # Each dataset must be handled in a custom way
    if dataset_name == "truthful_qa":
        raw_dataset = load_dataset(hf_path, name="generation")["validation"]
        for x in raw_dataset:
            yield Sample(
                instruction = x["question"],
                correct_answers = x["correct_answers"],
                incorrect_answers = x["incorrect_answers"],
            )
    else:
        raise NotImplementedError(f"{dataset_name} is not yet supported")

if __name__ == "__main__":
    args = parse_args()

    if args.seed:
        set_seed(args.seed)

    # Prepare output dir and file
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.dataset_name}.jsonl")
    f_out = open(out_path, "w")

    # Load data from HF
    input_data_iter = prepare_input_data_iter(args.dataset_name)
    for sample in tqdm(input_data_iter):
        # Determine models to be used to generate completions for each x
        sample.model_names = random.sample(MODEL_POOL, args.num_models)

        # Export sample
        print(sample.model_dump_json(), file=f_out, flush=True)

    f_out.close()
    