
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
    oracle_name: str = field(
        default="ultrafeedback",
        metadata={
            "help": "How the chosen and rejected responses should be identified.",
            "choices": ["random", "ultrafeedback"],
        },
    )
    acquisition_function: str = field(
        default="random",
        metadata={
            "help": "Acquisition function type",
            "choices": ["random", "dts", "infomax", "maxminlcb", "infogain"],
        },
    )
    reward_model: str | None = field(
        default=None,
        metadata={
            "help": "The reward model that should be trained via the loop.",
            "choices": ["enn"],
        }
    )

    # global configs
    seed: int = field(
        default=None, metadata={"help": "Random seed for reproducibility."}
    )
    max_length: int = field(
        default=4096, 
        metadata={"help": "Max length for the tokenizer."}
    )
    report_to: str | None = field(
        default=None,
        metadata={
            "help": "Reporting tool to use.",
            "choices": ["wandb", "tensorboard", "none"],
        },
    )
    debug: bool = field(
        default=False, 
        metadata={"help": "Set True when debugging the script for speed."},
    )
    outer_loop_batch_size: int = field(
        default=32, 
        metadata={"help": "Number of prompts that should be processed before reward trainer is called"},
    )
    compute_reward_batch_size: int = field(
        default=8,
        metadata={"help": "Number of completions per reward computation forward pass"},
    )
    save_every_n_outer_batches: int = field(
        default=100,
        metadata={"help": "How often (in outer loop batches) the intermediate dataset should be saved to disk."},
    )
    replay_buffer_size: int = field(
        default=3200,
        metadata={"help": "Size of the replay buffer that stores previous chosen/rejected pairs for training the reward model."},
    )

    timestamp: str | None = field(
        default=None,
        metadata={"help": "Timestamp at which this run was init."}
    )

    # active learning-related configs
    ## acquisition function
    acquisition_function_config: dict[str, dict] | None = field(
        default=None,
        metadata={
            "help": "Configs relevant for the chosen acquisition function",
        },
    )

    ## reward model
    reward_model_config: dict[str, dict] | None = field(
        default=None,
        metadata={
            "help": "Configs relevant for the possible reward models",
        },
    )
    reward_trainer_config: dict[str, dict] | None = field(
        default=None,
        metadata={
            "help": "Configs relevant for the possible reward trainers",
        },
    )

    # reproducibility-related configs
    logs_path: str | None = field(
        default=None, metadata={"help": "Path to save the logs for this script."}
    )
    args_path: str | None = field(
        default=None, metadata={"help": "Path to save the args for this script."}
    )
    wandb_dir: str | None = field(
        default=None, metadata={"help": "Path to local wandb records"}
    )
    wandb_project: str | None = field(
        default=None, metadata={"help": "WandB project name for logging."}
    )
    wandb_run_id: str | None = field(
        default=None, metadata={"help": "WandB run id for logging."}
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

# TODO: make more robust, this is ridiculous
def extract_annotator_name(dataset_path: str) -> str:
    for key in ["llama", "qwen"]:
        if key in path.basename(dataset_path):
            return key

def get_args() -> argparse.Namespace:
    parser = HfArgumentParser(LoopArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # load the YAML of values with which the namespace should be populated
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        for key, val in config.items():
            if hasattr(args, key):
                setattr(args, key, val)

    # create wandb configs that reflects acquisition function, annotator used to determine response quality, oracle that will determine chosen vs rejected, and current timestamp
    # similarly define output path, args path, logs path
    args.timestamp = get_timestamp(more_detailed=True)
    args.run_id = "_".join([
        args.acquisition_function,
        args.reward_model if args.reward_model is not None else "none",
        extract_annotator_name(args.inputs_path),
        args.oracle_name,
        args.timestamp,
    ])

    if args.report_to == "wandb":
        if args.reward_model is None:
            # because no reward model is involved, there is nothing to report to wandb
            args.report_to = None
        else:
            args.wandb_project = config["base_wandb_project"]
            args.wandb_dir = path.join(config["base_wandb_dir"], args.run_id)
            args.kpi_run_id = f"kpi_{args.run_id}"
            args.trainer_run_id = f"trainer_{args.run_id}"
    args.output_path = path.join(config["base_output_dir"], args.run_id)
    args.args_path = path.join(config["base_logs_dir"], f"{args.run_id}.args")
    args.logs_path = path.join(config["base_logs_dir"], f"{args.run_id}.log")

    try:
        reward_trainer_config = args.reward_trainer_config[args.reward_model]
        assert "output_dir" not in args.reward_trainer_config
        reward_trainer_config["output_dir"] = f"{config['base_trainer_dir']}/{args.run_id}"
    except:
        pass

    return args