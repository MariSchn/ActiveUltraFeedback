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
from transformers import HfArgumentParser, TrainerCallback
from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast_object_list
import wandb
import logging

from rewarduq.models.enn_reward_model import (
    ENNRewardModel,
    ENNRewardModelConfig,
    ENNRewardModelTrainer,
    ENNRewardModelTrainerConfig,
    ENNRewardModelPipeline,
    enn_compute_metrics,
)

from activeuf.acquisition_function import *
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
    --completions_dataset_path ${SCRATCH}/datasets/combined_annotations_llama/ \
    --output_path=$SCRATCH/datasets/testssss/ \
    --logs_path=$SCRATCH/logs_final_test \
    --args_path=$SCRATCH/models_enn_test \
    --acquisition_config=$SCRATCH/ActiveUltraFeedback/activeuf/acquisition_function/configs.yaml \
    --report_to="wandb" \
    --acquisition_function_type="double_thompson_sampling"
    
accelerate launch \
    --config_file=$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_gpu.yaml \
    -m activeuf.active_learning_loop \
    --completions_dataset_path ${SCRATCH}/datasets/combined_annotations_qwen/ \
    --output_path=$SCRATCH/datasets/testssss/ \
    --logs_path=$SCRATCH/logs_final_test \
    --args_path=$SCRATCH/models_enn_test \
    --acquisition_config=$SCRATCH/ActiveUltraFeedback/activeuf/acquisition_function/configs.yaml \
    --report_to="wandb" \
    --acquisition_function_type="double_thompson_sampling"

"""

# previous run stopped at 145 -th iteration.

# მადლობა, ძმაო, გაიხარე.
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_SILENT"] = "true"

# TODO:
# Take some of the arguments in active_learning_loop_config.yaml file. (To pass in some arguments more efficiently)


@dataclass
class LoopArguments:
    oracle_name: str = field(
        default="ultrafeedback",
        metadata={
            "help": "Type of oracle to use. Choices: ['random', 'ultrafeedback']"}
    )
    completions_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the full completions dataset."}
    )
    previous_output_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the dataset that is generated so far. These will be ignored in processing."}
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
        metadata={
            "help": "Acquisition function type. Choices: ['double_thompson_sampling', 'random', 'infomax', 'maxminlcb', 'infogain']"}
    )
    acquisition_config: str = field(
        default="activeuf/acquisition_function/configs.yaml",
        metadata={"help": "Acquisition function configuration file path."}
    )
    replay_buffer_size: int = field(
        default=3200,
        metadata={
            "help": "Size of the replay buffer for the ENN reward model training."}
    )
    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": "Reporting tool to use. Choices: ['wandb', 'tensorboard', 'none']"}
    )


global_step_offset = 0


def get_global_step_offset():
    return global_step_offset


# TODO: მოაშორე ეს აქსელერატორი აქედან და ზედმეტი პარამეტრები. რამე გასაღების ამოღება თუ გინდა, ამ კლასის საქმე არაა ეგ, გარედან ამოიღე და ისე გადააწოდე.
class WandbStepLoggerCallback(TrainerCallback):
    def __init__(self, step_offset_getter):
        self.step_offset_getter = step_offset_getter

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            absolute_step = self.step_offset_getter() + state.global_step
            if "loss_individual" in logs.keys():
                absolute_step += 1
            wandb.log(logs, step=absolute_step)


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
        print(
            f"Output path already exists, using {args.output_path} instead of {base_output_path}")

    if not args.logs_path:
        args.logs_path = f"logs/{args.timestamp}.log"

    if not args.args_path:
        args.args_path = f"logs/{args.timestamp}.args"

    if args.acquisition_function_type in ["random", "ultrafeedback"]:
        if args.report_to == "wandb":
            print(
                "Warning: WandB reporting is not supported for random or ultrafeedback acquisition functions.")
        args.report_to = None

    return args


def custom_collate_fn(batch):
    return {
        "prompt_id": [x["prompt_id"] for x in batch],
        "prompt": [x["prompt"] for x in batch],
        "source": [x["source"] for x in batch],
        "completions": [x["completions"] for x in batch],
    }


def acquisition_function_KPIs(rewards, chosen_idxs, rejected_idxs):
    """
        Function to calculate acquisition function KPIs.
        rewards: Tensor of shape (n_samples, n_completions, 2) - rewards for each completion
        chosen_idxs: Tensor of shape (n_samples, 1) - indices of the chosen completions
        rejected_idxs: Tensor of shape (n_samples, 1) - indices of the rejected completions

        list of KPIs to calculate:
        - mean rewards and mean uncertainties of:
        --- all completions per sample
        --- chosen completions per sample
        --- rejected completions per sample
        --- same as above, but for the whole batch

        TODO:
        track these KPIs for each model from the model pool separately.
        combined statistics for both chosen and rejected completions.
    """
    mean_rewards_per_sample = rewards.mean(dim=1)  # (n_samples, 2)
    mean_rewards_of_batch = mean_rewards_per_sample.mean(dim=0)  # (2,)

    chosen_rewards = rewards.gather(
        1, chosen_idxs.unsqueeze(-1).expand(-1, -1, rewards.size(-1))).squeeze(1)
    rejected_rewards = rewards.gather(
        1, rejected_idxs.unsqueeze(-1).expand(-1, -1, rewards.size(-1))).squeeze(1)

    mean_chosen_rewards = chosen_rewards.mean(dim=0)  # (2,)
    mean_rejected_rewards = rejected_rewards.mean(dim=0)  # (2,)

    # Add to KPIs
    kpis = {
        "mean_rewards_per_sample": mean_rewards_per_sample[:, 0].tolist(),
        "mean_rewards_per_batch": mean_rewards_of_batch[0].item(),
        "mean_uncertainty_per_sample": mean_rewards_per_sample[:, 1].tolist(),
        "mean_uncertainty_per_batch": mean_rewards_of_batch[1].item(),

        "mean_chosen_rewards_per_batch": mean_chosen_rewards[0].item(),
        "mean_chosen_uncertainty_per_batch": mean_chosen_rewards[1].item(),
        "mean_rejected_rewards_per_batch": mean_rejected_rewards[0].item(),
        "mean_rejected_uncertainty_per_batch": mean_rejected_rewards[1].item(),

        "chosen_rewards_per_sample": chosen_rewards[:, 0].tolist(),
        "chosen_uncertainty_per_sample": chosen_rewards[:, 1].tolist(),
        "rejected_rewards_per_sample": rejected_rewards[:, 0].tolist(),
        "rejected_uncertainty_per_sample": rejected_rewards[:, 1].tolist(),
    }
    return kpis


def acquisition_function_handler(acquisition_function_type):
    if acquisition_function_type == "double_thompson_sampling":
        max_iterations = acquisition_config.get("max_iterations", 10)
        beta = acquisition_config.get("beta", 1)
        # will be changed later.
        acquisition_function = DoubleThompsonSampling(
            beta=beta, max_iterations=max_iterations)
    elif acquisition_function_type == "random":
        acquisition_function = RandomAcquisitionFunction()
    elif acquisition_function_type == "infomax":
        acquisition_function = InfoMax()
    elif acquisition_function_type == "maxminlcb":
        beta = acquisition_config.get("beta", 1.0)
        argmax_tol = acquisition_config.get("argmax_tol", 1e-4)
        decision_buffer = acquisition_config.get("decision_buffer", 0.0)
        use_candidate_set = acquisition_config.get(
            "use_candidate_set", False)
        seed = acquisition_config.get("seed", 42)

        acquisition_function = MaxMinLCB(
            beta=beta,
            argmax_tol=argmax_tol,
            decision_buffer=decision_buffer,
            use_candidate_set=use_candidate_set,
            seed=seed
        )
    elif acquisition_function_type == "infogain":
        acquisition_function = InfoGain()
    else:
        raise ValueError(
            f"Unknown acquisition function type: {acquisition_function_type}")
    return acquisition_function


if __name__ == "__main__":
    accelerator = Accelerator()

    parser = HfArgumentParser((LoopArguments,))
    args, = parser.parse_args_into_dataclasses()
    args = parse_postprocess(args)

    # wandb init - only on main process
    if accelerator.is_main_process and args.report_to == "wandb":
        ENNTrainer_log_run = f"activeuf_enn_{args.timestamp}" + \
            ("llama" if "llama" in args.completions_dataset_path else "qwen")
        acquisition_KPI_run = f"activeuf_KPI_{args.timestamp}" + \
            ("llama" if "llama" in args.completions_dataset_path else "qwen")

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
    # setup(login_to_hf=True)

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
        logger.info(
            f"Filtering out {len(done_prompt_ids)} done samples from the data to be processed")
        dataset = dataset.filter(
            lambda x: x["prompt_id"] not in done_prompt_ids)
        output_dataset = done_dataset.to_list()
    else:
        output_dataset = []

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True,
    )

    logger.info(
        f"Creating acquisition function {args.acquisition_function_type}")
    acquisition_function = acquisition_function_handler(
        args.acquisition_function_type)

    logger.info(f"Creating oracle {args.oracle_name}")
    oracle = init_oracle(args.oracle_name)

    logger.info(f"Creating UQ model")
    # if args.acquisition_function_type in ["double_thompson_sampling", "infomax", "maxminlcb", "infogain"]:
    uq_pipeline = ENNRewardModelPipeline(
        ENNRewardModelConfig(
            # "meta-llama/Llama-3.2-1B-Instruct"
            base_model_name_or_path="unsloth/Qwen2.5-1.5B-Instruct"
        ),
        ENNRewardModelTrainerConfig(
            num_train_epochs=1,
            output_dir=f"trainer_output/{args.timestamp}",
            save_strategy="no",
            per_device_train_batch_size=math.ceil(
                args.batch_size / accelerator.num_processes),  # total will be exactly args.batch_size if: (B mod GPU_COUNT ≡ 0)
            report_to=None,  # * TEMPORARY: Disable logging to wandb
            disable_tqdm=True,
            logging_strategy="steps",
            logging_steps=1,
            run_name=f"activeuf_{args.timestamp}",
            lr_scheduler_type="constant",
            learning_rate=5e-6,
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
    } for _ in range(args.replay_buffer_size)]

    uq_pipeline.trainer = ENNRewardModelTrainer(
        args=uq_pipeline.trainer_config,
        model=uq_pipeline.model,
        processing_class=uq_pipeline.model.tokenizer,
        compute_metrics=enn_compute_metrics,
        train_dataset=Dataset.from_list(dummy_data)
    )
    if accelerator.is_main_process and args.report_to == "wandb":
        uq_pipeline.trainer.add_callback(
            WandbStepLoggerCallback(get_global_step_offset)
        )

    if args.previous_checkpoint_path:
        logger.info(f"Loading checkpoint from {args.previous_checkpoint_path}")
        uq_pipeline.model = ENNRewardModel.from_pretrained(
            args.previous_checkpoint_path)
        # # TODO:
        # load trainer state

    model = uq_pipeline.model
    tokenizer = model.tokenizer

    logger.info(f"Starting data generation loop")
    replay_buffer = deque(maxlen=args.replay_buffer_size)

    model, tokenizer = accelerator.prepare(model, tokenizer)

    total_processes = accelerator.num_processes

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

            # The input to the UQModel has to be partitioned, (upper limit is batch size of 256 + 80), if we input batch that is too big, we get CUDA_OUT_OF_MEMORY error.
            # has to be adjusted according to the ENN model's memory requirements, and GPU capacity
            microbatch_size = 8 * 16 + 8 * 10
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
                # Maybe you can use .extend()?
                rewards_list.append(mb_outputs["rewards"].cpu())

                del mb_inputs, mb_outputs
            torch.cuda.empty_cache()

            outputs = {"rewards": torch.cat(rewards_list, dim=0)}

            logger.info(
                f"- Uncertainty quantification took {end - start:.2f}s")

            # Select the completions that should be used for the binarized sample
            start = time.time()
            # (n_samples_in_batch, n_completions_per_sample, 3)
            rewards = outputs["rewards"].detach().view(
                n_samples_in_batch, -1, 3)
            # Replace last two columns with standard deviation (upper_bound - lower_bound) / 2
            # Shape: (n_samples_in_batch, n_completions_per_sample, 1)
            # rewards_mean = rewards[:, :, 0:1]
            # # Shape: (n_samples_in_batch, n_completions_per_sample, 1)
            # rewards_std = (rewards[:, :, 2:3] - rewards[:, :, 1:2]) / 2
            # # Shape: (n_samples_in_batch, n_completions_per_sample, 2)
            # rewards = torch.cat([rewards_mean, rewards_std], dim=-1)

            b_acquired_idxs = torch.tensor(                                                      # (n_samples_in_batch, 2)
                acquisition_function(*rewards.unbind(-1))
            )

            end = time.time()
            logger.info(f"- Acquisition function took {end - start:.2f}s")

            # (n_samples_in_batch, 2, max_length)
            temp = b_acquired_idxs.unsqueeze(-1).expand(-1, -
                                                        1, args.max_length)

            input_ids = inputs["input_ids"].cpu()
            b_acquired_input_ids = torch.take_along_dim(                                         # (n_samples_in_batch, 2, max_length)
                # (n_samples_in_batch, n_completions_per_sample, max_length)
                input_ids.view(n_samples_in_batch,
                               n_completions_per_sample, -1),
                temp,
                dim=1,
            )
            attention_masks = inputs["attention_mask"].cpu()
            b_acquired_attention_mask = torch.take_along_dim(                                    # (n_samples_in_batch, 2, max_length)
                # (n_samples_in_batch, n_completions_per_sample, max_length)
                attention_masks.view(n_samples_in_batch,
                                     n_completions_per_sample, -1),
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
                    # (max_length,)
                    "input_ids_1": b_acquired_input_ids[j, 0],
                    # (max_length,)
                    "attention_mask_1": b_acquired_attention_mask[j, 0],

                    "response_text_2": batch["completions"][b]["response_text"][j],
                    "model_2": batch["completions"][b]["model"][j],
                    "score_2": batch["completions"][b]["overall_score"][j],
                    # (max_length,)
                    "input_ids_2": b_acquired_input_ids[j, 1],
                    # (max_length,)
                    "attention_mask_2": b_acquired_attention_mask[j, 1],
                }
                for j, (a, b) in enumerate(b_acquired_idxs)
            ]

            # Call oracle to determine which is chosen and which is rejected
            annotated_batch_local = oracle(acquired_batch)

            chosen_idxs = []
            rejected_idxs = []
            for j, sample in enumerate(annotated_batch_local):
                if annotated_batch_local[j]["chosen_model"] == acquired_batch[j]["model_1"]:
                    chosen_idxs.append(b_acquired_idxs[j, 0].item())
                    rejected_idxs.append(b_acquired_idxs[j, 1].item())
                else:
                    chosen_idxs.append(b_acquired_idxs[j, 1].item())
                    rejected_idxs.append(b_acquired_idxs[j, 0].item())

            rewards_mean = rewards[:, :, 0:1]
            # Shape: (n_samples_in_batch, n_completions_per_sample, 1)
            rewards_std = (rewards[:, :, 2:3] - rewards[:, :, 1:2]) / 2
            # Shape: (n_samples_in_batch, n_completions_per_sample, 2)
            rewards_for_kpi = torch.cat([rewards_mean, rewards_std], dim=-1)

            acquisition_kpis = acquisition_function_KPIs(
                rewards_for_kpi, torch.tensor(chosen_idxs)[:, None], torch.tensor(
                    rejected_idxs)[:, None]
            )

            local_acquisition_KPIs_sample = [{}
                                             for _ in range(len(chosen_idxs))]
            local_acquisition_KPIs_batch = [{}]
            for k, v in acquisition_kpis.items():
                if isinstance(v, list):
                    for j, val in enumerate(v):
                        local_acquisition_KPIs_sample[j][k] = val
                else:
                    local_acquisition_KPIs_batch[0][k] = v
            for j in range(len(local_acquisition_KPIs_sample)):
                local_acquisition_KPIs_sample[j]["prompt_id"] = annotated_batch_local[j]["prompt_id"]

            accelerator.wait_for_everyone()
            annotated_batch = gather_object(annotated_batch_local)
            acquisition_kpis_sample = gather_object(
                local_acquisition_KPIs_sample)
            acquisition_kpis_batch = gather_object(
                local_acquisition_KPIs_batch)

            # Update dataset to be saved, then save to disk
            if accelerator.is_main_process:
                output_dataset.extend([
                    {
                        k: v
                        for k, v in x.items()
                        if not k.startswith("input_ids") and not k.startswith("attention_mask")
                    } for x in annotated_batch
                ])

                Dataset.from_list(output_dataset).save_to_disk(
                    args.output_path)

        # Keep only unique prompt_id entries
        annotated_batch = list(
            {x["prompt_id"]: x for x in annotated_batch}.values())
        acquisition_kpis_sample = list(
            {x["prompt_id"]: x for x in acquisition_kpis_sample}.values())

        acquisition_kpis_batch_copy = {
            k: 0 for k in acquisition_kpis_batch[0].keys()}
        for k in acquisition_kpis_batch[0].keys():
            for j in range(total_processes):
                acquisition_kpis_batch_copy[k] += acquisition_kpis_batch[j][k]
        for k in acquisition_kpis_batch_copy.keys():
            acquisition_kpis_batch_copy[k] /= len(acquisition_kpis_batch)
        acquisition_kpis_batch = acquisition_kpis_batch_copy

        if accelerator.is_main_process and args.report_to == "wandb":
            wandb.init(
                project="huggingface",
                name=acquisition_KPI_run,
                id=acquisition_KPI_run,
                resume=(i > 0),
                config=vars(args),
                allow_val_change=True,
            )
            for j, sample_kpis in enumerate(acquisition_kpis_sample):
                sample_kpis_no_id = {
                    k: v for k, v in sample_kpis.items() if k != "prompt_id"}
                wandb.log(sample_kpis_no_id, step=args.batch_size * i + j + 1)
            wandb.log(acquisition_kpis_batch, step=(i+1) * args.batch_size)
            wandb.finish()

        # Update replay buffer
        replay_buffer.extend(annotated_batch)

        if args.acquisition_function_type == "random":
            continue
        start = time.time()
        if accelerator.is_main_process and args.report_to == "wandb":
            wandb.init(
                project="huggingface",  # <-- replace with your wandb project
                name=ENNTrainer_log_run,
                id=ENNTrainer_log_run,
                resume=(get_global_step_offset() > 0),
                config=vars(args),
                allow_val_change=True,  # Allow changing the config values
            )

        model.train()

        uq_pipeline.trainer.train_dataset = Dataset.from_list(replay_buffer)
        if hasattr(uq_pipeline.trainer, "train_dataloader"):
            uq_pipeline.trainer.train_dataloader = None  # Force dataloader rebuild

        uq_pipeline.trainer.train()

        global_step_offset += uq_pipeline.trainer.state.global_step

        end = time.time()
        logger.info(f"- Training took {end - start:.2f}s")
        logger.info(f"Done with batch {i}\n")
        torch.cuda.empty_cache()
        if accelerator.is_main_process and args.report_to == "wandb":
            wandb.finish()
