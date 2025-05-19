import json
import argparse
import yaml
import os
import time
from collections import deque

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset

from rewarduq.models import ENNRewardModelPipeline, ENNRewardModelConfig, ENNRewardModelTrainerConfig

from activeuf.acquisition_function.acquisition import RandomAcquisitionFunction, DoubleThompsonSampling
from activeuf.oracle.oracles import init_oracle
from activeuf.utils import get_logger, setup, set_seed, get_timestamp
from activeuf.configs import *
from activeuf.schemas import *

"""
This script takes a dataset with completions as input and generate a binary preference dataset, determining the best completion (chosen/rejected) pair,
using an uncertainty quantification reward model, followed by an acquisition function, which determines which 2 completions should be selected for the oracle.
The oracle is then used to determine which completion is chosen and which is rejected.

Example run command:
    torchrun -m activeuf.active_learning_loop \
        --completions_dataset_path datasets/ultrafeedback_annotated \
        --output_size 100
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--oracle_name", type=str, default="ultrafeedback", help="Type of oracle to use.", choices=["random", "ultrafeedback"])

    parser.add_argument("--completions_dataset_path", type=str, required=True, help="Path to the completions dataset.")
    parser.add_argument("--output_size", type=int, default=None, help="Desired output size of the dataset. If not provided, the entire input dataset will be used")
    parser.add_argument("--output_path", type=str, help="Path to save the annotated dataset.")
    parser.add_argument("--logs_path", type=str, help="Path to save the logs for this script.")

    parser.add_argument("--batch_size", type=int, default=3, help="Batch Size for uncertainty sampling.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")
    
    parser.add_argument("--acquisition_function_type", type=str, default="double_thompson_sampling", help="Acquistion function type", choices=["double_thompson_sampling", "random"])
    parser.add_argument("--acquisition_config", type=str, default="activeuf/acquisition_function/acquisition_config.yaml", help="acquisition function configuration file path")
    parser.add_argument("--replay_buffer_size", type=int, default=20, help="Size of the replay buffer for the ENN reward model training.")
    args = parser.parse_args()

    if not args.output_path:
        args.output_path = f"{args.completions_dataset_path.rstrip('/')}-active-{get_timestamp()}"
    assert not os.path.exists(args.output_path), f"Output path {args.output_path} already exists"

    if not args.logs_path:
        args.logs_path = f"logs/{get_timestamp()}.log"

    return args

def custom_collate_fn(batch):
    return {
        "prompt_id": [x["prompt_id"] for x in batch],
        "prompt": [x["prompt"] for x in batch],
        "source": [x["source"] for x in batch],
        "completions": [x["completions"] for x in batch],
    }


if __name__ == "__main__":
    args = parse_args()
    logger = get_logger(__name__, args.logs_path)

    logger.info("Logging into HuggingFace")
    setup(login_to_hf=True)

    if args.seed:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    logger.info("Parsing config")
    try:
        # Attempt to load the reward configuration file
        with open(args.acquisition_config, "r") as f:
            acquisition_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: The specified reward configuration file '{args.acquisition_config}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse the acquisition configuration file '{args.acquisition_config}'.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading the acquisition configuration file: {e}")

    logger.info(f"Loading completions from {args.completions_dataset_path}")
    dataset = load_from_disk(args.completions_dataset_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=custom_collate_fn,
        shuffle=True, 
    )

    logger.info(f"Creating acquisition function {args.acquisition_function_type}")
    if args.acquisition_function_type == "double_thompson_sampling":
        max_iterations = acquisition_config.get("max_iterations", 10)
        beta = acquisition_config.get("beta", 1)
        acquisition_function = DoubleThompsonSampling(beta=beta, max_iterations=max_iterations) # will be changed later.
    elif args.acquisition_function_type == "random":
        acquisition_function = RandomAcquisitionFunction()
    else:
        raise ValueError(f"Unknown acquisition function type: {args.acquisition_function_type}")

    logger.info(f"Creating oracle {args.oracle_name}")
    oracle = init_oracle(args.oracle_name)

    logger.info(f"Creating UQ model")
    if args.acquisition_function_type == "double_thompson_sampling":
        # TODO: Pass correct arguments to the UQ model
        uq_pipeline = ENNRewardModelPipeline(
            ENNRewardModelConfig(
                base_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct"
            ),
            ENNRewardModelTrainerConfig(
                report_to="none"  # * TEMPORARY: Disable logging to wandb
            )
        )
    # * If the acquisition function is random, there is no need to calculate uncertainties

    logger.info(f"Starting data generation loop")
    replay_buffer = deque(maxlen=args.replay_buffer_size)
    output_dataset = []

    tokenizer = uq_pipeline.model.tokenizer
    for i, batch in enumerate(dataloader):
        logger.info(f"Processing batch {i}")

        start = time.time()
        n_samples = len(batch["prompt_id"])
        # Prepare batch to be input into the model
        all_messages = [
            [
                {
                    "role": "user", 
                    "content": batch["prompt"][sample_idx],
                },
                {
                    "role": "system", 
                    "content": completion["response_text"],
                } 
            ]
            for sample_idx in range(n_samples)
            for completion in batch["completions"][sample_idx]
        ]

        model_inputs = tokenizer.apply_chat_template(
            all_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        model_inputs = tokenizer(
            model_inputs, 
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ).to(uq_pipeline.model.device)
        end = time.time()
        logger.info(f"- Preprocessing took {end - start:.2f}s")

        # Get reward and uncertainty (lower and upper bounds)
        start = time.time()
        with torch.no_grad():
            outputs = uq_pipeline.model(**model_inputs)
        end = time.time()
        logger.info(f"- Uncertainty quantification took {end - start:.2f}s")

        temp = outputs["rewards"].detach().view(n_samples, -1, 3)
        rewards, lower_bounds, upper_bounds = temp[:,:,0], temp[:,:,1], temp[:,:,2]

        # Select the completions that should be used for the binarized sample
        acquisition_idxs = acquisition_function(
            rewards, lower_bounds, upper_bounds
        )
        acquired_batch = [
            {   
                "prompt_id": batch["prompt_id"][i],
                "source": batch["source"][i],
                "prompt": batch["prompt"][i],
                "response_text_1": batch["completions"][i][a]["response_text"],
                "model_1": batch["completions"][i][a]["model"],
                "overall_score_1": batch["completions"][i][a]["overall_score"],
                "response_text_2": batch["completions"][i][b]["response_text"],
                "model_2": batch["completions"][i][b]["model"],
                "overall_score_2": batch["completions"][i][b]["overall_score"],
            }
            for i, (a, b) in enumerate(acquisition_idxs)
        ]
        
        # Call oracle to determine which is chosen and which is rejected
        annotated_batch = oracle(acquired_batch)

        # Add the batch to the replay buffer
        output_dataset.extend(annotated_batch)
        replay_buffer.extend(annotated_batch)

        logger.info(f"Saving output dataset to {args.output_path}")
        start = time.time()
        os.makedirs(args.output_path, exist_ok=True)
        temp = Dataset.from_list(output_dataset)
        temp.save_to_disk(args.output_path)
        end = time.time()
        logger.info(f"- Saving took {end - start:.2f}s")

        # Train the reward model with the new data
        start = time.time()
        train_dataset = Dataset.from_list(list(replay_buffer))
        uq_pipeline.train(train_dataset)
        end = time.time()
        logger.info(f"- Training took {end - start:.2f}s")
        logger.info(f"Done with batch {i}\n")

        # Check if we have reached the desired output size
        if args.output_size and len(output_dataset) >= args.output_size:
            logger.info(f"Reached desired output size of {args.output_size}. Stopping data generation.")
            break

    args_path = os.path.join(args.output_path, "args.json")
    with open(args_path, "w") as f_out:
        json.dump(vars(args), f_out)