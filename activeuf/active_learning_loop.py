import json
import argparse
import yaml
import os
import time
import sys
from collections import deque
import math

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast_object_list

from rewarduq.models.enn_reward_model import (
    ENNRewardModel,
    ENNRewardModelConfig,
    ENNRewardModelTrainer,
    ENNRewardModelTrainerConfig,
    ENNRewardModelPipeline,
    enn_compute_metrics,
)

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

accelerate launch \
    --config_file=$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_gpu.yaml \
    -m activeuf.active_learning_loop \
    --completions_dataset_path ${SCRATCH}/datasets/ultrafeedback_annotated/ \
    --output_path=$SCRATCH/datasets/testssss/ \
    --logs_path=$SCRATCH/logs_final_test \
    --args_path=$SCRATCH/models_enn_test \
    --acquisition_config=/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/activeuf/acquisition_function/configs.yaml
"""

@dataclass
class LoopArguments:
    oracle_name: str = field(
        default="ultrafeedback",
        metadata={"help": "Type of oracle to use. Choices: ['random', 'ultrafeedback']"}
    )
    completions_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the full completions dataset."}
    )
    previous_output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dataset that is generated so far. These will be ignored in processing."}
    )
    previous_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model checkpoint."}
    )
    output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the annotated dataset."}
    )
    logs_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the logs for this script."}
    )
    args_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the args for this script."}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch Size for uncertainty sampling."}
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "Max length for the tokenizer."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."}
    )
    acquisition_function_type: str = field(
        default="double_thompson_sampling",
        metadata={"help": "Acquisition function type. Choices: ['double_thompson_sampling', 'random']"}
    )
    acquisition_config: str = field(
        default="activeuf/acquisition_function/configs.yaml",
        metadata={"help": "Acquisition function configuration file path."}
    )
    replay_buffer_size: int = field(
        default=3200,
        metadata={"help": "Size of the replay buffer for the ENN reward model training."}
    )

def parse_postprocess(args: argparse.Namespace) -> argparse.Namespace:
    args.timestamp = get_timestamp()

    if not args.output_path:
        args.output_path = f"{args.completions_dataset_path.rstrip('/')}-active-{args.timestamp}"
    base_output_path = args.output_path
    suffix = 2
    while os.path.exists(args.output_path):
        if base_output_path.endswith('/'):
            base_output_path = base_output_path.rstrip('/')
        args.output_path = f"{base_output_path}_{suffix}"
        suffix += 1
    if suffix > 2:
        print(f"Output path already exists, using {args.output_path} instead of {base_output_path}")

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
    accelerator = Accelerator()
    
    parser = HfArgumentParser((LoopArguments,))
    args, = parser.parse_args_into_dataclasses()
    args = parse_postprocess(args)

    # GPU cleanup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Setting everything up
    with open(args.args_path, "w") as f_out:
        json.dump(vars(args), f_out)

    logger = get_logger(__name__, args.logs_path)
    
    logger.info("--- ARGUMENTS ---")
    logger.info(args)

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
                base_model_name_or_path="unsloth/Qwen2.5-1.5B-Instruct"#"meta-llama/Llama-3.2-1B-Instruct"
            ),
            ENNRewardModelTrainerConfig(
                num_train_epochs=1,
                output_dir=f"trainer_output/{args.timestamp}",
                save_strategy="no",
                #per_device_train_batch_size=8,
                report_to="none",  # * TEMPORARY: Disable logging to wandb
                disable_tqdm=True,
            )
        )
        # Initialize the trainer with an empty Dataset having the required keys. So we have access to the uq_pipeline.trainer before entering the loop.
        dummy_data = [{
            'prompt': '',
            'prompt_id': '',
            'chosen': '',
            'chosen_model': '',
            'chosen_score': 0,
            'input_ids_chosen': [],
            'attention_mask_chosen': [],
            'rejected': '',
            'rejected_model': '',
            'rejected_score': 0,
            'input_ids_rejected': [],
            'attention_mask_rejected': []
        }]
        uq_pipeline.trainer = ENNRewardModelTrainer(
            args=uq_pipeline.trainer_config,
            model=uq_pipeline.model,
            processing_class=uq_pipeline.model.tokenizer,
            compute_metrics=enn_compute_metrics,
            train_dataset=Dataset.from_list(dummy_data)
        )
    
    if args.previous_checkpoint_path:
        logger.info(f"Loading checkpoint from {args.previous_checkpoint_path}")
        uq_pipeline.model = ENNRewardModel.from_pretrained(args.previous_checkpoint_path)
        #TODO:
        # load trainer state
    
    model = uq_pipeline.model
    tokenizer = model.tokenizer
    
    logger.info(f"Starting data generation loop")
    replay_buffer = deque(maxlen=args.replay_buffer_size)
    
    model, tokenizer = accelerator.prepare(model, tokenizer)
    
    
    total_processes = accelerator.num_processes
    current_process = accelerator.process_index
    for i, full_batch in enumerate(dataloader):
        batch_loader = DataLoader(Dataset.from_dict(full_batch), 
                                  batch_size=math.ceil(len(full_batch["prompt_id"])/total_processes))
        batch_loader = accelerator.prepare(batch_loader)
        
        for batch in batch_loader:
            logger.info(f"Processing batch {i}")

            start = time.time()
            n_samples_in_batch = len(batch["prompt_id"])
            n_completions_per_sample = len(batch["completions"])

            # Prepare messages for model
            messages = []
            for sample_idx in range(n_samples_in_batch):
                for completion in batch["completions"]:
                    messages.append([
                        {
                            "role": "user", 
                            "content": batch["prompt"][sample_idx],
                        },
                        {
                            "role": "system", 
                            "content": completion["response_text"][sample_idx],
                        } 
                    ])

            messages_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            inputs = tokenizer(
                messages_str, 
                padding="max_length",
                max_length=args.max_length,
                truncation=True,
                return_tensors="pt",
            ).to(model.device)   
            

            # inputs["input_ids]: (n_samples_in_batch * n_completions_per_sample, max_length)
            end = time.time()
            logger.info(f"- Preprocessing took {end - start:.2f}s")

            # Get reward and uncertainty (lower and upper bounds)
            start = time.time()
            model.eval()
            
            #The input to the UQModel has to be partitioned, (upper limit is batch size of 256 + 80), if we input batch that is too big, we get CUDA_OUT_OF_MEMORY error.
            microbatch_size = 8 * 16 + 8 * 10 #has to be adjusted according to the ENN model's memory requirements, and GPU capacity
            total = inputs["input_ids"].shape[0]

            rewards_list = []
            for mb_start in range(0, total, microbatch_size):
                mb_end = min(mb_start + microbatch_size, total)
                mb_inputs = {
                    "input_ids": inputs["input_ids"][mb_start:mb_end],
                    "attention_mask": inputs["attention_mask"][mb_start:mb_end]
                }
                with torch.no_grad():
                    mb_outputs = model(**mb_inputs)
                rewards_list.append(mb_outputs["rewards"].cpu()) #Maybe you can use .extend()?
                                        
                del mb_inputs, mb_outputs
            torch.cuda.empty_cache()
                                                    
            outputs = {"rewards": torch.cat(rewards_list, dim=0)}
            
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
            del inputs, total
            torch.cuda.empty_cache()
            
            acquired_batch = [
                {   
                    "prompt_id": batch["prompt_id"][j],
                    "source": batch["source"][j],
                    "prompt": batch["prompt"][j],

                    "response_text_1": batch["completions"][a]["response_text"][j],
                    "model_1": batch["completions"][a]["model"][j],
                    "score_1": batch["completions"][a]["overall_score"][j],
                    "input_ids_1": b_acquired_input_ids[j, 0],                                   # (max_length,)
                    "attention_mask_1": b_acquired_attention_mask[j, 0],                         # (max_length,)

                    "response_text_2": batch["completions"][b]["response_text"][j],
                    "model_2": batch["completions"][b]["model"][j],
                    "score_2": batch["completions"][b]["overall_score"][j],
                    "input_ids_2": b_acquired_input_ids[j, 1],                                   # (max_length,)
                    "attention_mask_2": b_acquired_attention_mask[j, 1],                         # (max_length,)
                }
                for j, (a, b) in enumerate(b_acquired_idxs)
            ]
            
            # Call oracle to determine which is chosen and which is rejected
            annotated_batch_local = oracle(acquired_batch)

            accelerator.wait_for_everyone()
            annotated_batch = gather_object(annotated_batch_local)            
            
            # Update dataset to be saved, then save to disk
            if accelerator.is_main_process:
                output_dataset.extend([
                    {
                        k: v 
                        for k, v in x.items()
                        if not k.startswith("input_ids") and not k.startswith("attention_mask")
                    } for x in annotated_batch
                ])
            
                Dataset.from_list(output_dataset).save_to_disk(args.output_path)

        # Keep only unique prompt_id entries
        annotated_batch = list({x["prompt_id"]: x for x in annotated_batch}.values())
        # Update replay buffer
        replay_buffer.extend(annotated_batch)        
        
        start = time.time()
        model.train()
        #check if Dataset.from_list(replay_buffer) is on GPU or CPU
        uq_pipeline.trainer.train_dataset = Dataset.from_list(replay_buffer)  # Update the trainer's dataset
        uq_pipeline.trainer.train()
        end = time.time()
        logger.info(f"- Training took {end - start:.2f}s")
        logger.info(f"Done with batch {i}\n")
        torch.cuda.empty_cache()