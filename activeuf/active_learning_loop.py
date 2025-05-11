import json
import argparse
import yaml
import os
import numpy as np
from tqdm import tqdm
from collections import deque

from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset

from rewarduq.models import ENNRewardModelPipeline, ENNRewardModelConfig, ENNRewardModelTrainerConfig

from activeuf.oracle.oracles import BaseOracle, RandomOracle, UltraFeedbackOracle
from activeuf.acquisition_function.acquisition import RandomAcquisitionFunction, DoubleThompsonSampling
from activeuf.utils import get_logger, setup, set_seed
from activeuf.configs import *
from activeuf.schemas import *

logger = get_logger(__name__)

"""
This script takes a dataset with completions as input and generate a binary preference dataset, determining the best completion (chosen/rejected) pair,
using an uncertainty quantification reward model, followed by an acquisition function, which determines which 2 completions should be selected for the oracle.
The oracle is then used to determine which completion is chosen and which is rejected.

Example run command:
    torchrun -m activeuf.active_learning_loop \
        --completion_dataset /iopsstor/scratch/cscs/smarian/datasets/allenai/ultrafeedback_binarized_cleaned/train_prefs-with-completions-sanitized \
        --output_size 100

    torchrun -m activeuf.active_learning_loop --completion_dataset /iopsstor/scratch/cscs/smarian/datasets/allenai/ultrafeedback_binarized_cleaned/train_prefs-with-completions-sanitized --output_size 100
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--oracle_type", type=str, default="ultrafeedback", help="Type of oracle to use.", choices=["random", "ultrafeedback"])

    parser.add_argument("--completion_dataset", type=str, required=True, help="Path to the prompt dataset.")
    parser.add_argument("--output_size", type=int, default=None, help="Desired output size of the dataset. If not provided, the entire input dataset will be used")
    parser.add_argument("--output_path", type=str, help="Path to save the annotated dataset.")

    parser.add_argument("--batch_size", type=int, default=3, help="Batch Size for uncertainty sampling.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")
    
    parser.add_argument("--acquisition_function_type", type=str, default="double_thompson_sampling", help="Acquistion function type", choices=["double_thompson_sampling", "random"])
    parser.add_argument("--acquisition_config", type=str, default="activeuf/acquisition_function/acquisition_config.yaml", help="acquisition function configuration file path")
    parser.add_argument("--replay_buffer_size", type=int, default=3200, help="Size of the replay buffer for the ENN reward model training.")
    args = parser.parse_args()

    if not args.output_path:
        args.output_path = f"{args.completion_dataset.rstrip('/')}-active"
    assert not os.path.exists(args.output_path), f"Output path {args.output_path} already exists"

    return args

if __name__ == "__main__":
    args = parse_args()

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

    logger.info(f"Loading Input Dataset {args.completion_dataset}")
    dataset = load_from_disk(args.completion_dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # TODO: Implement custom collate function to handle batching better. E.g. the role field of messages has length of batch_size, even though it should not have been batched

    logger.info(f"Creating acquisition function {args.acquisition_function_type}")
    if args.acquisition_function_type == "double_thompson_sampling":
        max_iterations = acquisition_config.get("max_iterations", 10)
        beta = acquisition_config.get("beta", 1)
        acquisition_function = DoubleThompsonSampling(beta=beta, max_iterations=max_iterations) # will be changed later.
    elif args.acquisition_function_type == "random":
        acquisition_function = RandomAcquisitionFunction()
    else:
        raise ValueError(f"Unknown acquisition function type: {args.acquisition_function_type}")

    logger.info(f"Creating oracle {args.oracle_type}")
    if args.oracle_type == "random":
        oracle = RandomOracle()
    elif args.oracle_type == "ultrafeedback":
        oracle = UltraFeedbackOracle()
    else:
        raise ValueError(f"Unknown oracle type: {args.oracle_type}")

    logger.info(f"Creating UQ model")
    if args.acquisition_function_type != "random":
        # * If the acquisition function is random, there is no need to calculate uncertainties
        # TODO: Pass correct arguments to the UQ model
        uq_pipeline = ENNRewardModelPipeline(
            ENNRewardModelConfig(
                base_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct"
            ),
            ENNRewardModelTrainerConfig(
                report_to="none"  # * TEMPORARY: Disable logging to wandb
            )
        )

    logger.info(f"Starting data generation loop")
    replay_buffer = deque(maxlen=args.replay_buffer_size)
    all_outputs = []

    num_completions = len(dataset[0]["completions"])
    tokenizer = uq_pipeline.model.tokenizer
    for batch in tqdm(dataloader, total=args.output_size // args.batch_size):
        # Prepare batch to be input into the model
        # TODO: Check if this can be done nicer, e.g. using a custom collate function
        all_messages = [
            [
                {
                    "role": "user", 
                    "content": batch["prompt"][sample_idx]
                },
                {
                    "role": "system", 
                    "content": batch["completions"][model_idx]["response_text"][sample_idx]
                } 
            ]
            for model_idx in range(num_completions)
            for sample_idx in range(args.batch_size)
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

        # Get reward and uncertainty (lower and upper bounds)
        outputs = uq_pipeline.model(**model_inputs)
        result = outputs["rewards"].detach().view(args.batch_size, -1, 3)

        reward, lower_bound, upper_bound = result[:, :, 0], result[:, :, 1], result[:, :, 2]

        # Select the completions that should be used for the binarized sample
        selected_idx = acquisition_function(reward, lower_bound, upper_bound)
        selected_completions = []
        for i, (idx_1, idx_2) in enumerate(selected_idx):
            # TODO: Check if there is a cleaner way to do this
            selected_completions.append({
                "prompt": batch["prompt"][i],
                "prompt_id": batch["prompt_id"][i],
                "completion_1": batch["completions"][idx_1]["response_text"][i],
                "model_1": batch["completions"][idx_1]["model"][0],
                "completion_2": batch["completions"][idx_2]["response_text"][i],
                "model_2": batch["completions"][idx_2]["model"][0],
            })
        
        # Call oracle to determine which is chosen and which is rejected
        annotated_batch = oracle(selected_completions)

        # Add the batch to the output dataset
        for sample in annotated_batch:
            replay_buffer.append(sample)
            all_outputs.append(sample)

        # Train the reward model with the new data
        output_dataset = Dataset.from_list(list(replay_buffer))
        uq_pipeline.train(output_dataset)

        # Check if we have reached the desired output size
        if args.output_size and len(all_outputs) >= args.output_size:
            logger.info(f"Reached desired output size of {args.output_size}. Stopping data generation.")
            break

    logger.info(f"Saving output dataset to {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    output_dataset = Dataset.from_list(all_outputs)
    output_dataset.save_to_disk(args.output_path)

    args_path = os.path.join(args.output_path, "args.json")
    with open(args_path, "w") as f_out:
        json.dump(vars(args), f_out)