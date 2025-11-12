import os
import argparse
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from datasets import load_dataset, load_from_disk, concatenate_datasets
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, get_peft_model
import pprint
import math
from accelerate import Accelerator
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR

os.environ["WANDB_PROJECT"] = "RM-Training"


def load_dataset_all(dataset_path):
    try:
        dataset = load_dataset(dataset_path)
    except Exception as e:
        try:
            dataset = load_from_disk(dataset_path)
        except Exception as e:
            print(f"Failed to load remote or local datasets: {e}")
            return
    return dataset


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

    if isinstance(train_dataset[0]["chosen"], str):
        train_dataset = train_dataset.map(
            lambda x:
                {
                    "chosen": [
                        {'content': x["prompt"], 'role': 'user'},
                        {'content': x["chosen"], 'role': 'assistant'}
                    ],
                    "rejected": [
                        {'content': x["prompt"], 'role': 'user'},
                        {'content': x["rejected"], 'role': 'assistant'}
                    ],
                }
        )

    # remove all columns except 'chosen' and 'rejected'
    train_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names if col not in ["chosen", "rejected"]]
    )

    return train_dataset


def train_reward_model(config, args):
    accelerator = Accelerator()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if accelerator.is_main_process:
        print("==== Training Arguments ====")
        print(args)
        print("==== YAML Configuration ====")
        pprint.pprint(config)

    output_dir = args.output_dir
    general_config = config.get("general", {})
    training_config = config.get("training", {})
    optimization_config = config.get("optimization", {})
    lr_scheduling_config = config.get("lr_scheduling", {})
    lora_config = config.get("lora", {})

    if accelerator.is_main_process and training_config.get("report_to", "none") == "wandb":
        print("wandb_dir changed")
        now = datetime.now()
        os.environ.setdefault(
            "WANDB_DIR", f"/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/wandb/job_{now.strftime('%Y%m%d-%H%M%S') + f'-{now.microsecond // 1000:03d}'}"
        )
        print("wandb directory is: ", os.environ["WANDB_DIR"])

    base_model = general_config.get(
        "base_model", "meta-llama/Llama-3.2-1B-Instruct")
    dataset_path = args.dataset_path
    if args.seed is None:
        seed = general_config.get("seed", 42)
    else:
        seed = args.seed

    print("Seed before setting: ", seed)
    set_seed(seed)
    torch_dtype = torch.bfloat16 if general_config.get(
        "torch_dtype", "bfloat16") == "bfloat16" else torch.float32

    # Load dataset
    dataset = load_dataset_all(dataset_path)
    dataset_2 = None
    if args.dataset_path_2 is not None:
        dataset_2 = load_dataset_all(args.dataset_path_2)
        print("Using the second dataset as well")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        print("No pad_token present. Assigning tokenizer.pad_token = tokenizer.eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        torch_dtype=torch_dtype,
        pad_token_id=tokenizer.pad_token_id,
    )

    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "SEQ_CLS"),
    )

    # Adjustments for PEFT model
    try:
        model = get_peft_model(model, peft_config)
    except Exception as e:
        print(f"Failed to apply PEFT model: {e}")
        print("Falling back to manually identifying target modules\n")
        target_modules = []
        for name, _ in model.named_modules():
            if any(p in name for p in lora_config.get("target_module_patterns", ["query", "key", "value"])):
                target_modules.append(name)
        peft_config.target_modules = target_modules
        model = get_peft_model(model, peft_config)

    trainer_config = RewardConfig(
        output_dir=output_dir,
        per_device_train_batch_size=math.ceil(training_config.get(
            "train_batch_size", 32) / accelerator.num_processes),
        per_device_eval_batch_size=math.ceil(training_config.get(
            "eval_batch_size", 32) / accelerator.num_processes),
        gradient_accumulation_steps=training_config.get("grad_acc_steps", 2),
        num_train_epochs=training_config.get("epochs", 1),
        learning_rate=float(optimization_config.get("learning_rate", 5e-6)),
        max_length=training_config.get("max_length", 4096),
        warmup_steps=lr_scheduling_config.get("num_warmup_steps", 10),
        logging_steps=training_config.get("logging_steps", 10),
        bf16=training_config.get("bf16", True),
        remove_unused_columns=training_config.get(
            "remove_unused_columns", False),
        report_to=training_config.get("report_to", "none"),
        save_strategy=training_config.get("save_strategy", "no"),
        save_steps=training_config.get("save_steps", 500),
        max_steps=training_config.get("max_steps", -1),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "linear"),
        run_name=os.path.basename(os.path.normpath(output_dir)),
        dataset_num_proc=accelerator.num_processes
    )

    if accelerator.is_main_process:
        print("==== Trainer Configuration ====")
        pprint.pprint(trainer_config)

    train_dataset = process_dataset(dataset)
    if dataset_2 is not None:
        train_dataset_2 = process_dataset(dataset_2)
        train_dataset = concatenate_datasets([train_dataset, train_dataset_2])

    if args.shuffle:
        train_dataset = train_dataset.shuffle(seed=seed)

    if accelerator.is_main_process:
        print("==== Dataset Columns ====")
        print(train_dataset.column_names)
        print("==== Dataset ====")
        print(train_dataset)
        print("==== Dataset Sample ====")
        print(train_dataset[0]["chosen"])
        print(train_dataset[0]["rejected"])

    if args.debug:
        train_dataset = train_dataset.select(range(270))

    if args.subset_size:
        train_dataset = train_dataset.select(
            range(min(len(train_dataset), args.subset_size)))

    print(f"doing training with {len(train_dataset)} samples")

    # Are you sure this doesn't internally shuffle the training dataset?
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=trainer_config,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset, Ignored for now.
    )

    trainer.train()

    # Save final model
    if trainer.is_world_process_zero():
        model = model.merge_and_unload()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")


# These utility functions could be moved to a separate file for better organization
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Reward model training using RewardTrainer.")
    parser.add_argument("--reward_config", required=True,
                        help="Path to the YAML reward training config.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save trained model.")
    parser.add_argument("--dataset_path", required=True,
                        help="Path to the dataset for training.")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode with a smaller dataset for testing.")
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Number of samples to use from the dataset.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for initialization.")
    parser.add_argument("--dataset_path_2", type=str, default=None,
                        help="Path to the second dataset for training.")
    parser.add_argument("--shuffle", action="store_true",
                        help="Whether to shuffle the dataset.")
    return parser.parse_args()


def load_config(config_path):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(
            f"Failed to load config: {e}\nUsing default configuration for training.\n")
        return {}


def main():
    args = parse_arguments()
    config = load_config(args.reward_config)
    train_reward_model(config, args)


if __name__ == "__main__":
    main()
