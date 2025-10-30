import argparse
from dataclasses import dataclass, field
import os.path as path
from transformers import HfArgumentParser
import yaml

from activeuf.acquisition_function.arguments import (
    RandomConfig,
    UltraFeedbackConfig,
    DTSConfig,
    IDSConfig,
    RUCBConfig,
)
from activeuf.utils import ensure_dataclass


@dataclass
class AcquisitionFunctionConfig:
    random: RandomConfig = field(
        metadata={"help": "Config for the random acquisition function."},
    )
    ultrafeedback: UltraFeedbackConfig = field(
        metadata={"help": "Config for the ultrafeedback acquisition function."},
    )
    dts: DTSConfig = field(
        metadata={"help": "Config for the dts acquisition function."},
    )
    ids: IDSConfig = field(
        metadata={"help": "Config for the ids acquisition function."},
    )
    rucb: RUCBConfig = field(
        metadata={"help": "Config for the rucb acquisition function."},
    )


@dataclass
class ENNModelConfig:
    base_model_name_or_path: str = field(
        metadata={"help": "Name or path of the base model."}
    )
    num_heads: int = field(
        metadata={"help": "The number of MLPs in the ensemble head."}
    )
    head_num_layers: int = field(
        metadata={"help": "The number of layers in each MLP of the ensemble head."}
    )
    head_hidden_dim: int = field(
        metadata={
            "help": "The dimension of the hidden layers in each MLP of the ensemble head."
        }
    )
    freeze_base_model: bool = field(
        metadata={"help": "Whether to freeze the base model during training."}
    )
    feature_extraction_layer: str = field(
        metadata={"help": "Which layer to use for feature extraction."}
    )
    head_initialization_xavier_gain: float = field(
        metadata={"help": "Xavier gain for weight initialization."}
    )


@dataclass
class ENNTrainerConfig:
    warmup_ratio: float = field()
    lr_scheduler_type: str = field(
        metadata={
            "help": "Type of learning rate scheduler.",
            "choices": ["constant", "linear", "cosine"],
        }
    )
    learning_rate: float = field(metadata={"help": "Initial learning rate."})
    num_train_epochs: int = field(
        metadata={"help": "Number of training epochs per outer batch added."}
    )
    regularization_towards_initial_weights: float = field(
        metadata={"help": "Initial regularization strength"}
    )
    max_length: int = field(metadata={"help": "Maximum sequence length."})
    center_rewards_coefficient: float | None = field(
        metadata={
            "help": "Coefficient to incentivize the reward model to output mean-zero rewards"
        }
    )
    precompute_features: bool = field()
    bf16: bool = field(metadata={"help": "Whether to use bfloat16 precision."})
    disable_tqdm: bool = field(metadata={"help": "Disable tqdm progress bars."})
    report_to: str = field(metadata={"help": "Reporting tool for the trainer."})
    save_strategy: str = field(metadata={"help": "Strategy for saving checkpoints."})
    save_steps: int = field(metadata={"help": "Number of steps between each save"})
    logging_strategy: str = field(metadata={"help": "Strategy for logging."})
    logging_steps: int = field(metadata={"help": "Number of steps between each log"})
    output_dir: str | None = field(
        default=None,
        metadata={"help": "Where to save checkpoints."},
    )


@dataclass
class ENNRegularizationConfig:
    initial_value: float = field(
        metadata={"help": "Strength of regularization towards initial weights."}
    )
    decay_type: str = field(
        metadata={
            "help": "Type of decay for regularization.",
            "choices": ["linear", "exponential"],
        }
    )
    exponential_decay_base: float = field(
        metadata={"help": "Base for exponential decay regularization."}
    )


@dataclass
class ENNConfig:
    previous_checkpoint_path: str | None = field(
        metadata={
            "help": "Path to a previous checkpoint if resuming training.",
        }
    )
    effective_batch_size: int = field(
        metadata={"help": "Effective batch size for training ENN."}
    )
    inference_batch_size: int = field(
        metadata={"help": "Number of completions per reward forward pass."}
    )

    model: ENNModelConfig = field(metadata={"help": "Configuration for the ENN model."})
    trainer: ENNTrainerConfig = field(
        metadata={"help": "Trainer configuration for ENN."}
    )
    regularization: ENNRegularizationConfig = field(
        metadata={"help": "Regularization settings for ENN."}
    )
    max_steps: int = field(
        metadata={"help": "Maximum number of training steps per outer batch added."}
    )

@dataclass
class LoopConfig:
    # dataset-related configs
    inputs_path: str = field(
        metadata={"help": "Path to the dataset with prompts and response texts."}
    )
    oracle_name: str = field(
        metadata={
            "help": "Oracle scorer for response texts.",
            "choices": ["random", "ultrafeedback"],
        }
    )
    acquisition_function_type: str = field(
        metadata={
            "help": "Acquisition function type",
            "choices": ["random", "dts", "infomax", "maxminlcb", "infogain"],
        }
    )
    reward_model_type: str = field(
        metadata={
            "help": "Reward model to train.",
            "choices": ["none", "enn"],
        }
    )

    # global configs
    seed: int = field(metadata={"help": "Random seed for reproducibility."})
    max_length: int = field(metadata={"help": "Max length for all sequences."})
    debug: bool = field(
        metadata={"help": "Set True when debugging the script for speed."}
    )
    outer_loop_batch_size: int = field(
        metadata={"help": "Number of prompts per outer loop batch."}
    )
    save_every_n_outer_batches: int = field(
        metadata={"help": "Save dataset every N outer loop batches."}
    )
    replay_buffer_factor: int = field(
        metadata={
            "help": "Replay buffer for reward model training will contain up to (replay_buffer_factor * outer_loop_batch_size) samples."
        }
    )

    # active learning-related configs
    acquisition_function: AcquisitionFunctionConfig = field(
        metadata={"help": "Configs for acquisition functions."}
    )
    enn: ENNConfig | None = field(
        metadata={"help": "All configs related to ENN reward model and training."}
    )

    # derived fields
    env_local_path: str = ""
    timestamp: str = ""
    run_id: str = ""
    output_path: str = ""
    args_path: str = ""
    logs_path: str = ""
    wandb_project: str = ""
    wandb_dir: str = ""


# TODO: make more robust, this is ridiculous
def extract_annotator_name(dataset_path: str) -> str:
    for key in ["llama", "qwen"]:
        if key in path.basename(dataset_path):
            return key


def get_loop_args(timestamp) -> argparse.Namespace:
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument(
        "--config_path", required=True, help="Path to the YAML config"
    )
    config_path = cli_parser.parse_args().config_path
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # define timestamp, then use it to create a run id
    config_dict['timestamp'] = timestamp
    config_dict['run_id'] = "_".join(
        [
            config_dict['acquisition_function_type'],
            config_dict['reward_model_type'],
            extract_annotator_name(config_dict['inputs_path']),
            config_dict['oracle_name'],
            config_dict['timestamp'],
        ]
    )

    # setup paths
    config_dict['output_path'] = path.join(config_dict["base_output_dir"], config_dict['run_id'])
    config_dict['args_path'] = path.join(config_dict["base_logs_dir"], f"{config_dict['run_id']}.args")
    config_dict['logs_path'] = path.join(config_dict["base_logs_dir"], f"{config_dict['run_id']}.log")
    if config_dict['reward_model_type'] != "none":
        config_dict['wandb_project'] = config_dict["base_wandb_project"]
        config_dict['wandb_dir'] = path.join(config_dict["base_wandb_dir"], config_dict['run_id'])

        trainer_args = config_dict[config_dict['reward_model_type']]["trainer"]
        if not trainer_args.get("output_dir"):
            trainer_args["output_dir"] = f"{config_dict['base_trainer_dir']}/{config_dict['run_id']}"

    parser = HfArgumentParser(LoopConfig)
    args = parser.parse_dict(config_dict, allow_extra_keys=True)[0]
    args = ensure_dataclass(LoopConfig, vars(args))

    return args
