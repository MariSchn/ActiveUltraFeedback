import json
import argparse
import yaml
import os
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset

from rewarduq.models import ENNRewardModelPipeline

from activeuf.oracle.oracles import BaseOracle, RandomOracle, UltraFeedbackOracle
from activeuf.acquisition_function.acquisition import RandomAcquisitionFunction, DoubleThompsonSampling
from activeuf.utils import get_logger, setup, set_seed
from activeuf.configs import *
from activeuf.schemas import *

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--uq_model_path", type=str, help="Path to uncertainty quantification model.")
    parser.add_argument("--uq_model_config", type=str, help="Path to uncertainty quantification config.")
    parser.add_argument("--uq_trainer_path", type=str, help="Path to uncertainty quantification trainer.")

    parser.add_argument("--completion_dataset", type=str, required=True, help="Path to the prompt dataset.")
    parser.add_argument("--output_size", type=int, default=None, help="Desired output size of the dataset. If not provided, the entire input dataset will be used")
    parser.add_argument("--output_path", type=str, help="Path to save the annotated dataset.")

    parser.add_argument("--num_iterations", type=int, default=10, help="Number of iterations in uncertainty sampling.")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch Size for uncertainty sampling.")
    
    parser.add_argument("--acquisition_function_type", type=str, default="double_thompson_sampling", help="Acquistion function type")
    parser.add_argument("--acquisition_config", type=str, default="activeuf/acquisition_function/acquisition_config.yaml", help="acquisition function configuration file path")
    args = parser.parse_args()

    if not args.output_path:
        args.output_path = f"{args.dataset_path.rstrip('/')}-active"
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
        print(f"Error: Failed to parse the reward configuration file '{args.reward_config}'.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading the reward configuration file: {e}")

    logger.info(f"Loading Input Dataset {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

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

    logger.info(f"Creating UQ model {args.uq_model_path}")
    if args.acquisition_function_type != "random":
        # * If the acquisition function is random, there is no need to calculate uncertainties
        # TODO: Pass correct arguments to the UQ model
        uq_pipeline = ENNRewardModelPipeline()

    logger.info(f"Starting data generation loop")
    output_dataset = Dataset.from_dict({
        "prompt": [],
        "chosen": [],
        "rejected": []
    })

    for batch in tqdm(dataloader, total=args.output_size // args.batch_size):
        # Get reward and uncertainty (lower and upper bounds)
        result = uq_pipeline(batch)
        reward, lower_bound, upper_bound = result[0, :], result[1, :], result[2, :]

        # Select the completions that should be used for the binarized sample
        selected_idx = acquisition_function(reward, lower_bound, upper_bound)
        selected_completions = None  # TODO

        # Call oracle to determine which is chosen and which is rejected
        labels = oracle(selected_completions)

        # Add the batch to the output dataset
        # TODO: Check if this is done properly
        output_dataset = output_dataset.add_item({
            "prompt": batch["prompt"],
            "chosen": None,
            "rejected": None
        })

        # Train the reward model with the new data
        uq_pipeline.train(output_dataset, labels)

    logger.info(f"Saving output dataset to {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    output_dataset.save_to_disk(args.output_path)

    args_path = os.path.join(args.output_path, "args.json")
    with open(args_path, "w") as f_out:
        json.dump(vars(args), f_out)