import json
import argparse
import yaml
import os
import time
from collections import deque

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset

from rewarduq.models import ENNRewardModel, ENNRewardModelPipeline, ENNRewardModelConfig, ENNRewardModelTrainerConfig

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
        --previous_output_path datasets/ultrafeedback_annotated-active-20250521-170158 \
        --previous_checkpoint_path trainer_output/20250521-170158/checkpoint-4
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--oracle_name", type=str, default="ultrafeedback", help="Type of oracle to use.", choices=["random", "ultrafeedback"])

    parser.add_argument("--completions_dataset_path", type=str, required=True, help="Path to the full completions dataset.")
    parser.add_argument("--previous_output_path", type=str, help="Path to the dataset that is generated so far. These will be ignored in processing.")
    parser.add_argument("--previous_checkpoint_path", type=str, help="Path to the reward model checkpoint.")

    parser.add_argument("--output_path", type=str, help="Path to save the annotated dataset.")
    parser.add_argument("--logs_path", type=str, help="Path to save the logs for this script.")
    parser.add_argument("--args_path", type=str, help="Path to save the args for this script.")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size for uncertainty sampling.")
    parser.add_argument("--max_length", type=int, default=1024, help="Max length for the tokenizer.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")
    
    parser.add_argument("--acquisition_function_type", type=str, default="double_thompson_sampling", help="Acquistion function type", choices=["double_thompson_sampling", "random"])
    parser.add_argument("--acquisition_config", type=str, default="activeuf/acquisition_function/configs.yaml", help="acquisition function configuration file path")
    parser.add_argument("--replay_buffer_size", type=int, default=32, help="Size of the replay buffer for the ENN reward model training.")
    args = parser.parse_args()

    args.timestamp = get_timestamp()

    if not args.output_path:
        args.output_path = f"{args.completions_dataset_path.rstrip('/')}-active-{args.timestamp}"
    assert not os.path.exists(args.output_path), f"Output path {args.output_path} already exists"

    if not args.logs_path:
        args.logs_path = f"logs/{args.timestamp}.log"

    if not args.args_path:
        args.args_path = f"logs/{args.timestamp}.args"

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
    with open(args.args_path, "w") as f_out:
        json.dump(vars(args), f_out)

    logger = get_logger(__name__, args.logs_path)

    logger.info("Logging into HuggingFace")
    setup(login_to_hf=True)

    if args.seed:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    logger.info("Parsing config")
    with open(args.acquisition_config, "r") as f:
        acquisition_config = yaml.safe_load(f)

    logger.info(f"Loading completions from {args.completions_dataset_path}")
    dataset = load_from_disk(args.completions_dataset_path)
    if args.previous_output_path:
        done_dataset = load_from_disk(args.previous_output_path)
        done_prompt_ids = set(done_dataset["prompt_id"])
        logger.info(f"Filtering out {len(done_prompt_ids)} done samples from the data to be processed")
        dataset = dataset.filter(lambda x: x["prompt_id"] not in done_prompt_ids)
        output_dataset = done_dataset.to_list()
    else:
        output_dataset = []
        
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
        uq_pipeline = ENNRewardModelPipeline(
            ENNRewardModelConfig(
                base_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct"
            ),
            ENNRewardModelTrainerConfig(
                num_train_epochs=1,
                output_dir=f"trainer_output/{args.timestamp}",
                save_strategy="epoch",
                report_to="none",  # * TEMPORARY: Disable logging to wandb
                disable_tqdm=True,
            )
        )
    if args.previous_checkpoint_path:
        logger.info(f"Loading checkpoint from {args.previous_checkpoint_path}")
        uq_pipeline.model = ENNRewardModel.from_pretrained(args.previous_checkpoint_path)

    model = uq_pipeline.model
    model = model.to("cuda")
    tokenizer = uq_pipeline.model.tokenizer

    logger.info(f"Starting data generation loop")
    replay_buffer = deque(maxlen=args.replay_buffer_size)

    for i, batch in enumerate(dataloader):
        logger.info(f"Processing batch {i}")

        start = time.time()
        n_samples_in_batch = len(batch["prompt_id"])
        n_completions_per_sample = len(batch["completions"][0])

        # Prepare messages for model
        messages = [
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
            for sample_idx in range(n_samples_in_batch)
            for completion in batch["completions"][sample_idx]
        ]

        messages_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        inputs = tokenizer(
            messages_str, 
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(uq_pipeline.model.device)                                                       # inputs["input_ids]: (n_samples_in_batch * n_completions_per_sample, max_length)
        end = time.time()
        logger.info(f"- Preprocessing took {end - start:.2f}s")

        # Get reward and uncertainty (lower and upper bounds)
        start = time.time()
        model.eval()
        with torch.no_grad():
            outputs = uq_pipeline.model(**inputs)                                            # output["rewards"]: (n_samples_in_batch * n_completions_per_sample, 3)
        end = time.time()
        logger.info(f"- Uncertainty quantification took {end - start:.2f}s")

        # Select the completions that should be used for the binarized sample
        start = time.time()
        rewards = outputs["rewards"].detach().view(n_samples_in_batch, -1, 3)                # (n_samples_in_batch, n_completions_per_sample, 3)
        b_acquired_idxs = torch.tensor(                                                      # (n_samples_in_batch, 2)
            acquisition_function(*rewards.unbind(-1))
        )
        end = time.time()
        logger.info(f"- Acquisition function took {end - start:.2f}s")

        temp = b_acquired_idxs.unsqueeze(-1).expand(-1, -1, args.max_length)                 # (n_samples_in_batch, 2, max_length)
        input_ids = inputs["input_ids"].cpu()
        b_acquired_input_ids = torch.take_along_dim(                                         # (n_samples_in_batch, 2, max_length)
            input_ids.view(n_samples_in_batch, n_completions_per_sample, -1),                # (n_samples_in_batch, n_completions_per_sample, max_length)
            temp, 
            dim=1,
        )
        attention_masks = inputs["attention_mask"].cpu()
        b_acquired_attention_mask = torch.take_along_dim(                                    # (n_samples_in_batch, 2, max_length)
            attention_masks.view(n_samples_in_batch, n_completions_per_sample, -1),          # (n_samples_in_batch, n_completions_per_sample, max_length)
            temp,
            dim=1,
        )

        acquired_batch = [
            {   
                "prompt_id": batch["prompt_id"][i],
                "source": batch["source"][i],
                "prompt": batch["prompt"][i],

                "response_text_1": batch["completions"][i][a]["response_text"],
                "model_1": batch["completions"][i][a]["model"],
                "score_1": batch["completions"][i][a]["overall_score"],
                "input_ids_1": b_acquired_input_ids[i, 0],                                   # (max_length,)
                "attention_mask_1": b_acquired_attention_mask[i, 0],                         # (max_length,)

                "response_text_2": batch["completions"][i][b]["response_text"],
                "model_2": batch["completions"][i][b]["model"],
                "score_2": batch["completions"][i][b]["overall_score"],
                "input_ids_2": b_acquired_input_ids[i, 1],                                   # (max_length,)
                "attention_mask_2": b_acquired_attention_mask[i, 1],                         # (max_length,)
            }
            for i, (a, b) in enumerate(b_acquired_idxs)
        ]

        # Call oracle to determine which is chosen and which is rejected
        annotated_batch = oracle(acquired_batch)

        # Update dataset to be saved, then save to disk
        output_dataset.extend([
            {
                k: v 
                for k, v in x.items()
                if not k.startswith("input_ids") and not k.startswith("attention_mask")
            } for x in annotated_batch
        ])
        Dataset.from_list(output_dataset).save_to_disk(args.output_path)

        # Update replay buffer
        replay_buffer.extend(annotated_batch)

        # Train UQ model on replay buffer
        start = time.time()
        model.train()
        uq_pipeline.train(Dataset.from_list(replay_buffer))
        end = time.time()
        logger.info(f"- Training took {end - start:.2f}s")
        logger.info(f"Done with batch {i}\n")
