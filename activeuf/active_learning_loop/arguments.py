
import argparse
from dataclasses import dataclass, field
import os.path as path
from transformers import HfArgumentParser
import yaml

from activeuf.utils import get_timestamp

@dataclass
class LoopArguments:

    config_path: str | None = field(
        default=None,
        metadata={"help": "Path to the YAML with the loop arguments."},
    )

    # dataset-related configs
    inputs_path: str | None = field(
        default=None,
        metadata={"help": "Path to the full completions dataset."}
    )
    base_output_path: str | None = field(
        default=None, metadata={"help": "Path to save the annotated dataset."}
    )
    oracle_name: str = field(
        default="ultrafeedback",
        metadata={
            "help": "How the chosen and rejected responses should be identified.",
            "choices": ["random", "ultrafeedback"],
        },
    )

    # active learning-related configs
    ## acquisition function-specific configs
    acquisition_function: str = field(
        default="random",
        metadata={
            "help": "Acquisition function type",
            "choices": ["random", "dts", "infomax", "maxminlcb", "infogain"],
        },
    )
    acquisition_function_config: dict | None = field(
        default=None,
        metadata={
            "help": "Configs relevant for the chosen acquisition function",
        },
    )

    ## reward model-related configs
    base_model_name_or_path: str | None = field(
        default=None, metadata={"help": "Base model name or path for the reward model."}
    )
    max_length: int = field(
        default=2048, 
        metadata={"help": "Max length for the tokenizer."}
    )

    ## reward model training configs
    max_training_steps: int = field(
        default=128, metadata={"help": "Maximum number of training steps."}
    )
    outer_loop_batch_size: int = field(
        default=64, metadata={"help": "Number of prompts to involve per reward model training call."}
    )
    rm_training_batch_size: int = field(
        # Must be equal to the total number of GPUs, otherwise OOM ERRORS! Change it to 4 when running on a single node.
        default=8,
        metadata={"help": "Number of preference samples to train on simultaneously."},
    )
    replay_buffer_size: int = field(
        default=8,
        metadata={
            "help": "Supplement the new set of preference samples with this many previous preference samples, and train on them simultaneously."
        },
    )
    use_features: bool = field(
        default=False,
        metadata={
            "help": "Whether to precompute or load cached last layer embeddings of the model and use them."
        },
    )

    # reproducibility-related configs
    seed: int = field(
        default=None, metadata={"help": "Random seed for reproducibility."}
    )
    logs_path: str | None = field(
        default=None, metadata={"help": "Path to save the logs for this script."}
    )
    args_path: str | None = field(
        default=None, metadata={"help": "Path to save the args for this script."}
    )
    report_to: str | None = field(
        default=None,
        metadata={
            "help": "Reporting tool to use.",
            "choices": ["wandb", "tensorboard", "none"],
        },
    )
    wandb_project: str | None = field(
        default=None, metadata={"help": "WandB project name for logging."}
    )

    # configs for starting from a previous point to save time 
    previous_checkpoint_path: str | None = field(
        default=None, metadata={"help": "Path to the reward model checkpoint."}
    )
    previous_output_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to the dataset that is generated so far. These will be ignored in processing."
        },
    )

    # misc configs
    debug: bool = field(
        default=False, 
        metadata={"help": "Set True when debugging the script for speed."},
    )

def extract_annotator_name(dataset_path: str) -> str:
    return path.basename(dataset_path.rstrip("/")).split("_")[-1]

def get_args() -> argparse.Namespace:
    parser = HfArgumentParser(LoopArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # load the YAML of values with which the namespace should be populated
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    for key, val in config.items():
        setattr(args, key, val)

    # create output path that reflects acquisition function, annotator used to determine response quality, oracle that will determine chosen vs rejected, and current timestamp
    # similarly define args and logs paths
    timestamp = get_timestamp(more_detailed=True)
    args.output_path = path.join(
        config["base_output_dir"], 
        "_".join([
            args.acquisition_function,
            extract_annotator_name(args.inputs_path),
            args.oracle_name,
            timestamp,
    ]))
    args.args_path = path.join(config["base_logs_dir"], f"{args.output_path}.args")
    args.logs_path = path.join(config["base_logs_dir"], f"{args.output_path}.log")

    # set wandb project name based on acquisition function
    if args.report_to == "wandb":
        if args.acquisition_function in ["random", "ultrafeedback"]:
            print(
                f"Warning: WandB reporting is not supported for acquisition_function={args.acquisition_function}."
            )
            # TODO: ask davit why these lines
            # args.max_length = 1
            # args.outer_loop_batch_size = 8192
            # enn["base_model_name_or_path"] = "unsloth/Qwen2.5-1.5B-Instruct"
        else:
            args.wandb_project = f"{args.base_wandb_project}_{args.acquisition_function}"

    return args