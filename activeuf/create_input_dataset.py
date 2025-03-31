import argparse
import os
import random
from tqdm import tqdm

from datasets import load_dataset

from activeuf.schemas import Sample
from activeuf.configs import DATASET_POOL, MODEL_POOL
from activeuf.utils import set_seed

"""
This script downloads a dataset from HuggingFace and processes it to follow the input data schema (defined in `schemas.py`) for the ActiveUltraFeedback pipeline.
The output of this script is a jsonl dataset WITHOUT LLM completions (which are to be added by the main script `main_vllm.py` later).

Example run command from project root:
    python -m create_input_dataset --dataset_name truthful_qa
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to download and process (e.g. truthful_qa)")
    
    parser.add_argument("--num_models", type=int, default=4, help="The number of models to use for completions for each sample")
    parser.add_argument("--seed", type=int, default=123, help="Seed for random sampling")

    parser.add_argument("--output_dir", type=str, default="../datasets/input_datasets/", help="The directory for exporting the input dataset")
    return parser.parse_args()

def load_input_data(dataset_name: str) -> list[dict]:
    # Given a supported dataset name, load the dataset from HF and extract the relevant fields
    if dataset_name not in DATASET_POOL:
        raise ValueError(f"{dataset_name} must be added to DATASET_POOL")
    
    if dataset_name == "truthful_qa":
        raw_dataset = load_dataset("truthfulqa/truthful_qa", name="generation")["validation"]
        input_data = [{
            "instruction": x["question"],
            "correct_answers": x["correct_answers"],
            "incorrect_answers": x["incorrect_answers"],
        } for x in raw_dataset]
    else:
        raise NotImplementedError(f"{dataset_name} is not yet supported")

    return input_data

if __name__ == "__main__":
    args = parse_args()

    if args.seed:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data from HF
    input_data = load_input_data(args.dataset_name)

    out_path = os.path.join(args.output_dir, f"{args.dataset_name}.jsonl")
    with open(out_path, "w") as f_out:
        for x in tqdm(input_data):
            # Determine models to be used to generate completions for each x
            x["model_names"] = random.sample(MODEL_POOL, args.num_models)

            # Sanity check: ensure x complies with Sample schema
            sample = Sample(**x)

            # Export sample
            # TODO: make sure this exports with double inverted commas instead
            print(sample.model_dump(mode="json"), file=f_out)