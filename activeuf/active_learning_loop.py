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
import random

from rewarduq.models.reward_head_ensemble import (
    RewardHeadEnsembleModel as ENNRewardModel,
    RewardHeadEnsembleModelConfig as ENNRewardModelConfig,
    RewardHeadEnsembleTrainer as ENNRewardModelTrainer,
    RewardHeadEnsembleTrainerConfig as ENNRewardModelTrainerConfig,
    RewardHeadEnsemblePipeline as ENNRewardModelPipeline,
    # enn_compute_metrics,
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
    --report_to="wandb" \
    --acquisition_function_type="dts"

accelerate launch \
    --config_file=$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_gpu.yaml \
    -m activeuf.active_learning_loop \
    --completions_dataset_path ${SCRATCH}/datasets/combined_annotations_qwen/ \
    --output_path=$SCRATCH/datasets/testssss/ \
    --report_to="wandb" \
    --acquisition_function_type="dts"
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
    outer_loop_batch_size: int = field(
        default=None,
        metadata={"help": "Batch Size for uncertainty sampling."}
    )
    rm_training_batch_size: int = field(
        # Must be equal to the total number of GPUs, otherwise OOM ERRORS! Change it to 4 when running on a single node.
        default=None,
        metadata={"help": "Batch Size for the ENN reward model training."}
    )
    max_length: int = field(
        default=None,
        metadata={"help": "Max length for the tokenizer."}
    )
    seed: int = field(
        default=None,
        metadata={"help": "Random seed for reproducibility."}
    )
    acquisition_function_type: str = field(
        default=None,
        metadata={
            "help": "Acquisition function type. Choices: ['dts', 'random', 'infomax', 'maxminlcb', 'infogain']"}
    )
    replay_buffer_size: int = field(
        default=None,
        metadata={
            "help": "Size of the replay buffer for the ENN reward model training."}
    )
    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": "Reporting tool to use. Choices: ['wandb', 'tensorboard', 'none']"}
    )
    loop_config: str = field(
        default="activeuf/active_learning_loop_config.yaml",
        metadata={"help": "Active learning loop configuration file path."}
    )
    debug: bool = field(
        default=False,
        metadata={"help": "Enable debugging mode."}
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "WandB project name for logging."}
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
    # Load the main config YAML
    with open(args.loop_config, "r") as f:
        config = yaml.safe_load(f)

    loop = config.get("loop", {})
    acquisition = config.get("acquisition", {})
    enn = config.get("enn", {})

    for key, value in loop.items():
        if hasattr(args, key):
            if getattr(args, key) is None:
                setattr(args, key, value)

    if args.acquisition_function_type is None:
        setattr(args, "acquisition_function_type",
                acquisition["acquisition_function_type"])

    # Acquisition function type + annotator model + timestamp
    args.timestamp = args.acquisition_function_type + "_" + \
        ("llama" if "llama" in args.completions_dataset_path else "qwen") + \
        "_" + get_timestamp()

    if not args.output_path:
        args.output_path = f"{args.completions_dataset_path.rstrip('/')}_active_{args.timestamp}"

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

    if not args.wandb_project:
        args.wandb_project = args.acquisition_function_type

    if args.acquisition_function_type in ["random", "ultrafeedback"]:
        if args.report_to == "wandb":
            print(
                "Warning: WandB reporting is not supported for random or ultrafeedback acquisition functions.")
        args.report_to = None
        args.max_length = 1
        args.outer_loop_batch_size = 8192
        enn["base_model_name_or_path"] = "unsloth/Qwen2.5-1.5B-Instruct"

    # Load acquisition config file and add as a dict
    acq_config_path = acquisition["acquisition_config"]
    if acq_config_path:
        with open(acq_config_path, "r") as f:
            args.acquisition_config = yaml.safe_load(f)
    else:
        args.acquisition_config = {}

    # Add enn_config as a dict
    args.enn_config = config.get("enn", {})

    return args


def custom_collate_fn(batch):
    return {
        "prompt_id": [x["prompt_id"] for x in batch],
        "prompt": [x["prompt"] for x in batch],
        "source": [x["source"] for x in batch],
        "completions": [x["completions"] for x in batch],
        "row_id": [x["row_id"] for x in batch],
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


def acquisition_function_handler(acquisition_function_type, acquisition_config):
    if acquisition_function_type == "dts":
        max_iterations = acquisition_config.get("max_iterations", 30)
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
        argmax_tol = float(acquisition_config.get("argmax_tol", 1e-4))
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
    elif acquisition_function_type == "relative_upper_confidence_bound":
        acquisition_function = RelativeUpperConfidenceBound()
    elif acquisition_function_type == "information_directed_sampling":
        acquisition_function = InformationDirectedSampling()
    elif acquisition_function_type == "ultrafeedback":
        acquisition_function = UltraFeedback()
    else:
        raise ValueError(
            f"Unknown acquisition function type: {acquisition_function_type}")
    return acquisition_function


if __name__ == "__main__":
    accelerator = Accelerator()

    # with TorchMemoryProfiler(record_mem_history=True, autosave=f"/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/activeuf/memhst5/memory_history{accelerator.process_index}"):
    parser = HfArgumentParser((LoopArguments,))
    args, = parser.parse_args_into_dataclasses()
    args = parse_postprocess(args)
    acquisition_config = args.acquisition_config
    enn_config = args.enn_config

    if accelerator.is_main_process:
        print("args: ", args)
        print("acquisition_config: ", acquisition_config)
        print("enn_config: ", enn_config)

    # wandb init - only on main process
    if accelerator.is_main_process and args.report_to == "wandb":
        ENNTrainer_log_run = f"loop_enn_{args.timestamp}"
        acquisition_KPI_run = f"loop_KPI_{args.timestamp}"

    # GPU cleanup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Setting everything up
    with open(args.args_path, "w") as f_out:
        json.dump(vars(args), f_out)

    logger = get_logger(__name__, args.logs_path)

    logger.info("--- ARGUMENTS ---")
    logger.info("args = %s", args)
    logger.info("acquisition_config = %s", acquisition_config)
    logger.info("enn_config = %s", enn_config)

    logger.info("Logging into HuggingFace")
    # setup(login_to_hf=True)

    if args.seed:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    logger.info(
        f"Loading completions from {args.completions_dataset_path}")
    dataset = load_from_disk(args.completions_dataset_path)
    dataset = dataset.add_column("row_id", list(range(len(dataset))))
    if args.debug:
        dataset = dataset.select(range(256))

    # dataset = dataset.select(range(args.outer_loop_batch_size))
    # Unfortunately the set of prompts have duplicate prompt_ids, so we can not filter by prompt_ids.
    # we have to make an assumption that the processed rows in the done dataset happened in the same order as would have happened on the original dataset. (Therefore we should not shuffle the dataset)
    # TODO: The loader below doesn't work.
    # - initial dataset was shuffled, so I have no order information
    # - I can not filter by prompt_id, because there are duplicates.
    # This is doable, but requires a bit of processing.
    if args.previous_output_path:
        done_dataset = load_from_disk(args.previous_output_path)
        done_prompt_count = len(done_dataset)
        dataset = dataset.select(range(done_prompt_count, len(dataset)))
        output_dataset = done_dataset.to_list()
    else:
        output_dataset = []

    dataloader = DataLoader(
        dataset,
        batch_size=args.outer_loop_batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True,
    )

    logger.info(f"Total number of samples in the dataset: {len(dataset)}")

    logger.info(
        f"Creating acquisition function {args.acquisition_function_type}")
    acquisition_function = acquisition_function_handler(
        args.acquisition_function_type, acquisition_config)
    logger.info("acquisition function class: %s",
                acquisition_function.__class__.__name__)

    logger.info(f"Creating oracle {args.oracle_name}")
    oracle = init_oracle(args.oracle_name)
    logger.info("oracle class: %s", oracle.__class__.__name__)

    logger.info(f"Creating UQ model")
    # if args.acquisition_function_type in ["dts", "infomax", "maxminlcb", "infogain"]:
    uq_pipeline = ENNRewardModelPipeline(
        ENNRewardModelConfig(
            base_model_name_or_path=enn_config.get(
                "base_model_name_or_path", "unsloth/Qwen2.5-1.5B-Instruct"),
            freeze_base_model=enn_config.get("freeze_base_model", True),
            feature_extraction_layer=enn_config.get(
                "feature_extraction_layer", "last_hidden_state"),
            # feature_extraction_selection_strategy="last",
            # fp16=True,
        ),
        ENNRewardModelTrainerConfig(
            num_train_epochs=enn_config.get("num_train_epochs", 1),
            output_dir=f"trainer_output/{args.timestamp}",
            save_strategy=enn_config.get("save_strategy", "no"),
            per_device_train_batch_size=math.ceil(
                args.rm_training_batch_size / accelerator.num_processes),  # total will be exactly args.batch_size if: (B mod GPU_COUNT ≡ 0)
            report_to=enn_config.get("report_to", "none"),
            disable_tqdm=enn_config.get("disable_tqdm", True),
            logging_strategy=enn_config.get("logging_strategy", "steps"),
            logging_steps=enn_config.get("logging_steps", 1),
            run_name=f"activeuf_{args.timestamp}",
            lr_scheduler_type=enn_config.get("lr_scheduler_type", "constant"),
            learning_rate=float(enn_config.get("learning_rate", 5e-6)),
            max_length=None,  # args.max_length,
            bf16=True,
            regularization_towards_initial_weights=enn_config.get(
                "regularization_towards_initial_weights", 10),
            # precompute_base_model_features=True,
            # precomputed_base_model_features_path="temp/",
            # eval_on_start=False,
            # max_steps=100,
        )
    )
    logger.info(
        f"batch size per gpu is {uq_pipeline.trainer_config.per_device_train_batch_size}")
    # Initialize the trainer with an empty Dataset having the required keys. So we have access to the uq_pipeline.trainer before entering the loop.
    dummy_data = [{
        'prompt': '',
        'prompt_id': '',
        'row_id': -1,
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
        train_dataset=Dataset.from_list(dummy_data)
    )
    if accelerator.is_main_process and args.report_to == "wandb":
        uq_pipeline.trainer.add_callback(
            WandbStepLoggerCallback(get_global_step_offset)
        )

    if args.previous_checkpoint_path:
        logger.info(
            f"Loading checkpoint from {args.previous_checkpoint_path}")
        uq_pipeline.model = ENNRewardModel.from_pretrained(
            args.previous_checkpoint_path)
        # # TODO:
        # load trainer state
    logger.info("UQ model class: %s", uq_pipeline.model.__class__.__name__)
    logger.info("UQ model config: %s", uq_pipeline.model_config)
    logger.info("UQ model tokenizer: %s",
                uq_pipeline.model.tokenizer.__class__.__name__)
    logger.info("UQ trainer class: %s", uq_pipeline.trainer.__class__.__name__)
    logger.info("UQ pipeline trainer config: %s", uq_pipeline.trainer_config)
    logger.info("UQ trainer args: %s", uq_pipeline.trainer.args)

    model = uq_pipeline.model
    tokenizer = model.tokenizer

    logger.info(f"Starting data generation loop")
    replay_buffer = deque(maxlen=args.replay_buffer_size)

    total_processes = accelerator.num_processes
    logger.info(f"Total processes: {total_processes}")

    if accelerator.is_main_process:
        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    initial_lambda_regularizer = float(
        uq_pipeline.trainer.args.regularization_towards_initial_weights)

    for i, full_batch in enumerate(dataloader):
        batch_loader = DataLoader(Dataset.from_dict(full_batch),
                                  batch_size=math.ceil(len(full_batch["prompt_id"])/total_processes))
        batch_loader = accelerator.prepare(batch_loader)

        annotated_batch = []
        acquisition_kpis_sample = []
        acquisition_kpis_batch = []
        # By default this loop is executed once. That is batch_loader.batch_size == len(batch_loader)
        logger.info(f"Processing batch {i}")

        for batch in batch_loader:

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

            if args.acquisition_function_type in ["random", "ultrafeedback"]:
                # Dummy values
                dummy_shape = (n_samples_in_batch *
                               n_completions_per_sample, args.max_length)
                device = model.device if 'model' in locals() else torch.device("cpu")
                inputs = {
                    "input_ids": torch.zeros(dummy_shape, dtype=torch.long, device=device),
                    "attention_mask": torch.zeros(dummy_shape, dtype=torch.long, device=device),
                }
            else:
                messages_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
                inputs = tokenizer(
                    messages_str,
                    padding="longest",
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

            if args.acquisition_function_type not in ["random", "ultrafeedback"]:
                # The input to the UQModel has to be partitioned, (upper limit is batch size of 256 + 80), if we input batch that is too big, we get CUDA_OUT_OF_MEMORY error.
                # has to be adjusted according to the ENN model's memory requirements, and GPU capacity
                # partitioning makes sense, because we have 17 completions per prompt. so around 18 * Batch_size / GPU_COUNT number of samples to process.
                # TODO: This parameter has to be tuned for each max_length parameter. Would be good to dynamically adjust, with a while true and try-except clause.
                # I don't understand why 8 has the quickest finish time (matte rof 0.5 seconds but still), also why do I have to split the input to the inference model,
                # why does it get memory issues if I don't split the data, makes no sense...
                microbatch_size = 8  # 8 * 16 + 8 * 10
                total = inputs["input_ids"].shape[0]

                rewards_list = []
                for mb_start in range(0, total, microbatch_size):
                    mb_end = min(mb_start + microbatch_size, total)
                    mb_inputs = {
                        "input_ids": inputs["input_ids"][mb_start:mb_end],
                        "attention_mask": inputs["attention_mask"][mb_start:mb_end]
                    }
                    with torch.no_grad():
                        mb_outputs = model(
                            **mb_inputs)
                    # Maybe you can use .extend()?
                    rewards_list.append(mb_outputs["rewards"].cpu())

                    del mb_inputs, mb_outputs
                    # maybe I have to write cuda.empty_cache() here
                del total
                torch.cuda.empty_cache()

                outputs = {"rewards": torch.cat(rewards_list, dim=0)}

            end = time.time()
            logger.info(
                f"- Uncertainty quantification took {end - start:.2f}s")

            # Select the completions that should be used for the binarized sample
            start = time.time()
            # (n_samples_in_batch, n_completions_per_sample, 3)
            if args.acquisition_function_type in ["ultrafeedback", "random"]:
                n_samples_in_batch = len(batch["prompt_id"])
                n_completions_per_sample = len(batch["completions"])
                rewards = torch.zeros(
                    (n_samples_in_batch, n_completions_per_sample, 3), dtype=torch.float32)
                for j in range(n_completions_per_sample):
                    # batch["completions"][j]["overall_score"] is a list of length n_samples_in_batch
                    rewards[:, j, 0] = torch.tensor(
                        batch["completions"][j]["overall_score"], dtype=torch.float32)

            else:
                rewards = outputs["rewards"].detach().view(
                    n_samples_in_batch, -1, 3)

            b_acquired_idxs = torch.tensor(                                                      # (n_samples_in_batch, 2)
                acquisition_function(*rewards.unbind(-1))
            )

            end = time.time()
            logger.info(f"- Acquisition function took {end - start:.2f}s")

            # (n_samples_in_batch, 2, max_length)
            temp = b_acquired_idxs.unsqueeze(-1).expand(-1, -
                                                        1, inputs["input_ids"].shape[-1])

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
            del inputs
            torch.cuda.empty_cache()

            acquired_batch = [
                {
                    "prompt_id": batch["prompt_id"][j],
                    "source": batch["source"][j],
                    "prompt": batch["prompt"][j],
                    "row_id": batch["row_id"][j],
                    "batch_id": i,

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
            rewards_for_kpi = torch.cat(
                [rewards_mean, rewards_std], dim=-1)

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
                local_acquisition_KPIs_sample[j]["row_id"] = annotated_batch_local[j]["row_id"]

            accelerator.wait_for_everyone()
            annotated_batch_tmp = gather_object(annotated_batch_local)
            acquisition_kpis_sample_tmp = gather_object(
                local_acquisition_KPIs_sample)
            acquisition_kpis_batch_tmp = gather_object(
                local_acquisition_KPIs_batch)

            annotated_batch.extend(annotated_batch_tmp)
            acquisition_kpis_sample.extend(acquisition_kpis_sample_tmp)
            acquisition_kpis_batch.extend(acquisition_kpis_batch_tmp)

        annotated_batch = list(
            {str(x["prompt_id"])+"_"+str(x["row_id"].item()): x for x in annotated_batch}.values())
        acquisition_kpis_sample = list(
            {str(x["prompt_id"])+"_"+str(x["row_id"].item()): x for x in acquisition_kpis_sample}.values())

        for x in acquisition_kpis_sample:
            if "row_id" in x:
                del x["row_id"]

        # Update dataset to be saved, then save to disk
        if accelerator.is_main_process:
            output_dataset.extend([
                {
                    k: v
                    for k, v in x.items()
                    if not k.startswith("input_ids") and not k.startswith("attention_mask")
                } for x in annotated_batch
            ])

            print(
                f"Saving {len(output_dataset)} samples to {args.output_path}")
            Dataset.from_list(output_dataset).save_to_disk(
                args.output_path)

        # processing the acquisition KPIs
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
                project=args.wandb_project,
                name=acquisition_KPI_run,
                id=acquisition_KPI_run,
                resume=(i > 0),
                config=vars(args),
                allow_val_change=True,
            )
            for j, sample_kpis in enumerate(acquisition_kpis_sample):
                sample_kpis_no_id = {
                    k: v for k, v in sample_kpis.items() if k != "prompt_id"}
                wandb.log(sample_kpis_no_id,
                          step=args.outer_loop_batch_size * i + j + 1)
            wandb.log(acquisition_kpis_batch, step=(
                i+1) * args.outer_loop_batch_size)
            wandb.finish()

        if args.acquisition_function_type in ["random", "ultrafeedback"]:
            continue

        # Update replay buffer
        replay_buffer.extend(annotated_batch)

        del batch_loader
        torch.cuda.empty_cache()
        start = time.time()
        if accelerator.is_main_process and args.report_to == "wandb":
            wandb.init(
                project=args.wandb_project,
                name=ENNTrainer_log_run,
                id=ENNTrainer_log_run,
                resume=(get_global_step_offset() > 0),
                config=vars(args),
                allow_val_change=True,  # Allow changing the config values
            )

        updated_lambda_regularizer = initial_lambda_regularizer * \
            args.outer_loop_batch_size / len(output_dataset)
        uq_pipeline.trainer.args.regularization_towards_initial_weights = updated_lambda_regularizer

        logger.info(
            f"Updated lambda regularizer: {updated_lambda_regularizer}")

        # Training
        model.train()
        # I need to perform max_training_steps steps.
        # I have rm_training_batch_size samples per step.
        # So I will have to sample max_training_steps * rm_training_batch_size samples.
        subset_size = min(
            len(replay_buffer), args.rm_training_batch_size * enn_config.get("max_training_steps", 10))
        random_subset = random.sample(
            replay_buffer, subset_size)

        uq_pipeline.trainer.train_dataset = Dataset.from_list(
            random_subset)

        if hasattr(uq_pipeline.trainer, "train_dataloader"):
            uq_pipeline.trainer.train_dataloader = None  # Force dataloader rebuild

        logger.info(
            f"Starting Training of the UQ model on {len(random_subset)} samples from the replay buffer")
        uq_pipeline.trainer.train()

        global_step_offset += uq_pipeline.trainer.state.global_step

        end = time.time()
        logger.info(f"- Training took {end - start:.2f}s")
        logger.info(f"Done with batch {i}\n")
        torch.cuda.empty_cache()
        if accelerator.is_main_process and args.report_to == "wandb":
            wandb.finish()
