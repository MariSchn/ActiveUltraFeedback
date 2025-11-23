import argparse
import os
import random
import re
import yaml
import torch
import numpy as np

# from trl import DPOConfig
from trl.data_utils import apply_chat_template, extract_prompt
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, load_dataset
from pprint import pprint

from accelerate import Accelerator

from activeuf.utils import set_seed, setup
from activeuf.dpo.trainer import NormedDPOConfig, NormedDPOTrainer

import wandb

"""
run command example:
accelerate launch --num_processes=4 \
    --config_file=$SCRATCH/ActiveUltraFeedback/configs/accelerate/multi_node.yaml -m activeuf.dpo.training \
    --config_path $SCRATCH/ActiveUltraFeedback/configs/dpo_training.yaml \
    --slurm_job_id $SLURM_JOB_ID \
    --dataset_path allenai/ultrafeedback_binarized_cleaned \
    --beta 0.1 
    
accelerate launch \
    --config_file=$SCRATCH/ActiveUltraFeedback/configs/accelerate/multi_node.yaml -m activeuf.dpo.training \
    --config_path $SCRATCH/ActiveUltraFeedback/configs/dpo_training.yaml \
    --slurm_job_id $SLURM_JOB_ID \
    --dataset_path /iopsstor/scratch/cscs/dmelikidze/datasets/active/dts_qwen_rgl100.0_wdcb0.995_obs128_rbs12800_1008156 
"""


# TODO: slurm_job_id is redundant, it is stored inside os.environ['SLURM_JOB_ID'] already.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="Path to the YAML training config."
    )
    parser.add_argument(
        "--slurm_job_id", type=str, help="SLURM Job ID associated with this run"
    )
    parser.add_argument("--dataset_path", type=str, help="Path to the training dataset")
    parser.add_argument("--beta", type=float, help="DPO beta parameter", default=0.1)
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate for training", required=False
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility", required=False
    )
    parser.add_argument(
        "--num_epochs", type=int, help="Number of training epochs", required=False
    )
    parser.add_argument(
        "--ablation_studies",
        action="store_true",
        help="Whether to run ablation studies",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        help="Base output directory for model checkpoints and logs",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for saving models",
        required=False,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode with a smaller dataset",
    )
    return parser.parse_args()


def process_dataset(dataset):
    # Dataset processing (Determining splits, restructuring (chosen/rejected columns))
    if isinstance(dataset, dict):
        if "train_prefs" in dataset:
            train_dataset = dataset["train_prefs"]
        elif "train" in dataset:
            train_dataset = dataset["train"]
        else:
            # TODO: More general way of handling dataset splits (They should come as arguments, for example).
            raise Exception(
                "Unknown dataset format. Expected 'train' or 'train_prefs' split."
            )
    else:
        train_dataset = dataset

    return train_dataset


def dataname_handler(dir_name, has_seeds=True):
    parts = dir_name.split("/")
    name_we_need = ""
    if has_seeds:
        judge_index = -4
        if "llama_3.3_70b" in parts[judge_index]:
            name_we_need += "llama70B_"
        elif "qwen_3_235b" in parts[judge_index]:
            name_we_need += "qwen235B_"
        elif "rm" in parts[judge_index]:
            name_we_need += "rm8Bsky_"
        else:
            raise ValueError("Unknown judge model in path: ", parts[judge_index])

        prompt_index = -5
        if "ultrafeedback_with_small" in parts[prompt_index]:
            name_we_need += "allenai_"
        elif "skywork_with_small" in parts[prompt_index]:
            name_we_need += "skywork_"
        elif "combined_with_small" in parts[prompt_index]:
            name_we_need += "combined_"
        else:
            raise ValueError("Unknown prompt source in path: ", parts[prompt_index])

        name_we_need += parts[-1] + "_"
        name_we_need += parts[-2]

    else:
        judge_index = -2
        if "llama_3.3_70b" in parts[judge_index]:
            name_we_need += "llama70B_"
        elif "qwen_3_235b" in parts[judge_index]:
            name_we_need += "qwen235B_"
        elif "rm" in parts[judge_index]:
            name_we_need += "rm8Bsky_"
        else:
            raise ValueError("Unknown judge model in path: ", parts[judge_index])

        prompt_index = -3
        if "ultrafeedback_with_small" in parts[prompt_index]:
            name_we_need += "allenai_"
        elif "skywork_with_small" in parts[prompt_index]:
            name_we_need += "skywork_"
        elif "combined_with_small" in parts[prompt_index]:
            name_we_need += "combined_"
        else:
            raise ValueError("Unknown prompt source in path: ", parts[prompt_index])

        name_we_need += parts[-1]

    return name_we_need


if __name__ == "__main__":
    accelerator = Accelerator()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # load env vars, args, configs
    setup()
    args = parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config["slurm_job_id"] = args.slurm_job_id
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.seed:
        config["seed"] = args.seed
    if args.num_epochs:
        config["training"]["num_train_epochs"] = args.num_epochs
    if args.base_output_dir:
        config["base_output_dir"] = args.base_output_dir

    lora_config = config.get("lora", {})
    training_config = config.get("training", {})

    def sanitize_name(name):
        # Replace / and . with _
        return re.sub(r"[/.]", "-", name)

    if args.ablation_studies:
        dataset_base = dataname_handler(
            args.dataset_path.rstrip("/"), has_seeds=(100 <= args.seed <= 104)
        )
        # print(dataset_base)
        # dataset_base = os.path.basename(args.dataset_path.rstrip("/"))
        # dataset_base = sanitize_name(dataset_base)

        # print(dataset_base)
        # prepare output dir based on SLURM job id and run name
        run_name = f"{args.slurm_job_id}-{dataset_base}"
    else:
        run_name = f"{args.slurm_job_id}-{sanitize_name(os.path.basename(args.dataset_path.rstrip('/')))}"

    # print(run_name)
    # exit()
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(config["base_output_dir"], run_name)
    if accelerator.is_main_process:
        print(f"Output dir: {output_dir}")
        print(f"Run name: {run_name}")
        print(f"Dataset: {args.dataset_path}")
        print("==== Training Arguments ====")
        print(args)
        print("==== YAML Configuration ====")
        pprint(config)

    # set seed for reproducibility
    if isinstance(config.get("seed"), int):
        set_seed(config.get("seed"))

    # send config file to wandb
    if accelerator.is_main_process and training_config.get("report_to") == "wandb":
        wandb.init(name=run_name)
        wandb.config.update(config)
        artifact = wandb.Artifact(run_name, type="config")
        artifact.add_file(args.config_path)
        wandb.log_artifact(artifact)

    # let all ranks wait here so that W&B is ready before training starts
    accelerator.wait_for_everyone()

    # Export config file for reproducibility
    out_path = os.path.join(output_dir, os.path.basename(args.config_path))

    if config.get("torch_dtype", "bfloat16") == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # load dataset, remove problematic columns
    print(f"Loading dataset from {args.dataset_path}...\n")
    dataset_path = args.dataset_path
    # Load dataset
    try:
        dataset = load_dataset(dataset_path)
    except Exception:
        try:
            dataset = load_from_disk(dataset_path)
        except Exception as e:
            print(f"Failed to load remote or local datasets: {e}")
            exit(-1)

    dataset = process_dataset(dataset)

    for column in ["messages", "prompt"]:
        try:
            dataset = dataset.remove_columns(column)
        except Exception:
            print(f"Unable to remove {column=} from dataset")

    # limit dataset if in debug mode
    if args.debug:
        dataset = dataset.select(range(100))

    # load tokenizer, then use it to remove overly long samples
    model_path = config["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # remove samples where prompt+chosen or prompt+rejected exceeds max length
    if training_config["max_length"]:
        # Do not load from cache here to avoid race conditions when running on multiple GPUs/nodes
        temp = dataset.map(
            extract_prompt,
            load_from_cache_file=False,
        )
        temp = temp.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": tokenizer},
            keep_in_memory=True,
            load_from_cache_file=False,
        )
        temp = temp.map(
            lambda _: NormedDPOTrainer.tokenize_row(
                _,
                processing_class=tokenizer,
                max_prompt_length=None,
                max_completion_length=None,
                add_special_tokens=True,
            ),
            load_from_cache_file=False,
        )

        old_n = len(dataset)
        print(f"Original number of samples: {old_n}")

        def check_if_short(x: dict) -> dict[str, bool]:
            return {
                "is_short": len(x["prompt_input_ids"]) + len(x["chosen_input_ids"])
                <= training_config["max_length"]
                or len(x["prompt_input_ids"]) + len(x["rejected_input_ids"])
                <= training_config["max_length"]
            }

        temp = temp.map(
            check_if_short,
            load_from_cache_file=False,
        )
        idxs = [i for i, _ in enumerate(temp["is_short"]) if _]
        dataset = dataset.select(idxs)
        print(
            f"Number of samples removed due to length constraints: {old_n - len(dataset)}"
        )

    dataset = dataset.select_columns(["chosen", "rejected"])

    # Manual shuffling of the dataset:
    dataset = dataset.shuffle(seed=config.get("seed"))

    if accelerator.is_main_process:
        print(dataset[0]["chosen"][0])

    # create lora version of model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    peft_config = LoraConfig(**lora_config)
    try:
        model = get_peft_model(model, peft_config)
    except Exception as e:
        print(f"Failed to apply PEFT model: {e}")
        print("Falling back to manually identifying target modules\n")
        # re-raise to avoid silent failures unless you want to implement further fallbacks
        raise

    # create DPO trainer
    trainer_config = NormedDPOConfig(
        run_name=run_name,
        output_dir=output_dir,
        dataset_num_proc=accelerator.num_processes,
        **training_config,
    )
    # DPO config, dataset loading. it doesnt know if tis evals or training.
    # we should shuffle.
    trainer = NormedDPOTrainer(
        model=model,
        args=trainer_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    print(trainer.normalize_logps)
    trainer.train()

    # Save final model
    if trainer.is_world_process_zero():
        model = model.merge_and_unload()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f_out:
            yaml.dump(config, f_out, default_flow_style=False)

    if (
        accelerator.is_main_process
        and training_config.get("report_to") == "wandb"
        and wandb.run is not None
    ):
        wandb.finish()
