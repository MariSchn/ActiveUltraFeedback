import argparse
import os
import shutil
import yaml

from trl import DPOConfig, DPOTrainer
from trl.data_utils import apply_chat_template, extract_prompt
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, load_dataset

from accelerate import Accelerator

from activeuf.configs import *
from activeuf.utils import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        help="Path to the YAML training config.")
    parser.add_argument("--slurm_job_id", type=str,
                        help="SLURM Job ID associated with this run")
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
                "Unknown dataset format. Expected 'train' or 'train_prefs' split.")
    else:
        train_dataset = dataset

    return train_dataset


if __name__ == "__main__":
    # load env vars, args, configs
    setup()
    args = parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    config["slurm_job_id"] = args.slurm_job_id
    lora_config = config.get("lora", {})
    training_config = config.get("training", {})

    # prepare output dir based on SLURM job id and run name
    run_name = f"{args.slurm_job_id}-{config['base_run_name']}"
    output_dir = os.path.join(config["base_output_dir"], run_name)

    # set seed for reproducibility
    if isinstance(config.get("seed"), int):
        set_seed(config.get("seed"))

    # prepare accelerator and torch stats
    accelerator = Accelerator()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if config.get("torch_dtype", "bfloat16") == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # load dataset, remove problematic columns
    print(f"Loading dataset from {config["train_path"]}...\n")
    dataset_path = config["train_path"]
    # Load dataset
    try:
        dataset = load_dataset(dataset_path)
    except Exception as e:
        try:
            dataset = load_from_disk(dataset_path)
        except Exception as e:
            print(f"Failed to load remote or local datasets: {e}")
            exit(-1)

    dataset = process_dataset(dataset)

    try:
        dataset = dataset.remove_columns(["messages"])
    except:
        pass

    # limit dataset if in debug mode
    if config.get("debug"):
        dataset = dataset.select(range(5000))

    # load tokenizer, then use it to remove overly long samples
    model_path = config["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # make dataset suitable for DPO training # this part is based on https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L617
    dataset = dataset.map(extract_prompt)
    dataset = dataset.map(apply_chat_template, fn_kwargs={
                          "tokenizer": tokenizer})

    # remove samples where prompt+chosen or prompt+rejected exceeds max length
    if training_config["max_length"]:
        dataset = dataset.map(lambda _: DPOTrainer.tokenize_row(
            _,
            processing_class=tokenizer,
            max_prompt_length=None,
            max_completion_length=None,
            add_special_tokens=True,
        ))

        old_n = len(dataset)
        print(f"Original number of samples: {old_n}")
        dataset = dataset.filter(
            lambda x: len(x["prompt_input_ids"]) + len(x["chosen_input_ids"]) <= training_config["max_length"] or
            len(x["prompt_input_ids"]) +
            len(x["rejected_input_ids"]) <= training_config["max_length"]
        )
        print(f"Number of samples removed: {old_n - len(dataset)}")

    # create lora version of model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch_dtype)
    peft_config = LoraConfig(**lora_config)
    try:
        model = get_peft_model(model, peft_config)
    except Exception as e:
        print(f"Failed to apply PEFT model: {e}")
        print("Falling back to manually identifying target modules\n")

        target_modules = []
        if "llama" in model_path.lower():
            for name, _ in model.named_modules():
                if any(_ in name for _ in ["query", "key", "value"]):
                    target_modules.append(name)
            peft_config.target_modules = target_modules
        else:
            print(
                f"Target module patterns for {model_path} must be implemented manually")
            raise

    # create DPO trainer
    trainer_config = DPOConfig(
        run_name=run_name, output_dir=output_dir, dataset_num_proc=accelerator.num_processes, **training_config)
    trainer = DPOTrainer(
        model=model,
        args=trainer_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    # Save final model
    if trainer.is_world_process_zero():
        model = model.merge_and_unload()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")

    # Export config file for reproducibility
    shutil.copy2(
        args.config_path,
        os.path.join(output_dir, os.path.basename(args.config_path)),
    )
