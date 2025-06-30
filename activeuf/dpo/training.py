import argparse
import os
import json
import yaml
import wandb

from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk

from activeuf.configs import *
from activeuf.utils import *

"""
This script is designed to train a (SFT) model using direct preference optimization (DPO).
To train the DPO model a (previously SFT trained) model is required and a binarized dataset.
The binarized dataset is a dataset that has been preprocessed to contain pairs of examples, where one example is preferred over the other.

Example run command:
    accelerate launch --config_file=./activeuf/dpo/accelerate_config.yml -m activeuf.dpo.training --model_path allenai/Llama-3.1-Tulu-3-8B-SFT --dataset_path /iopsstor/scratch/cscs/smarian/datasets/ultrafeedback_annotated_random --config ./activeuf/dpo/dpo_config.yml --debug

TODO: Add config file to avoid too many command line arguments
TODO: Add more customizable (hyper)parameters (e.g. learning rate, batch size, etc.). Only do after config file is added.
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train_path", type=str, help="The path to the local dataset to use for training (e.g. trl-lib/ultrafeedback_binarized/train)")
    parser.add_argument("--test_path", type=str, help="The path to the local dataset to use for testing (e.g. data/trl-lib/ultrafeedback_binarized/test)")
    parser.add_argument("--dataset_path", type=str, help="The HuggingFace path of the dataset to use for training and testing (e.g. trl-lib/ultrafeedback_binarized)")
    parser.add_argument("--config", type=str, help="The path to the YAML training config.")

    parser.add_argument("--model_path", type=str, help="The HuggingFace path of the model to train (e.g. Qwen/Qwen2-0.5B-Instruct)")

    parser.add_argument("--seed", type=int, default=123, help="Seed for random sampling")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode (just one train step on a small subset of the dataset)")

    parser.add_argument("--output_dir", type=str, default="models/dpo/", help="Where to export the tokenizer, trained model, and training args")
    return parser.parse_args()

def load_config(config_path):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {e}\nUsing default configuration for training.\n")
        return {}


if __name__ == "__main__":
    args = parse_args()

    setup(login_to_hf=True, login_to_wandb=True)

    # Set random seed
    if args.seed:
        set_seed(args.seed)

    config = load_config(args.config) if args.config else {}
    general_config = config.get("general", {})
    training_config = config.get("training", {})
    optimization_config = config.get("optimization", {})
    lr_scheduling_config = config.get("lr_scheduling", {})
    lora_config = config.get("lora", {})

    # Load the dataset
    print(f"Loading dataset from {args.dataset_path}...\n")
    dataset = load_from_disk(args.dataset_path) if os.path.exists(args.dataset_path) else load_dataset(args.dataset_path)["train_prefs"]

    if "messages" in dataset.column_names and all(k in dataset.column_names for k in ["prompt", "chosen", "rejected"]):
        print("Warning: Dataset contains 'messages' column along with 'prompt', 'chosen', 'rejected'. Removing 'messages' column to avoid conflict.")
        dataset = dataset.remove_columns(["messages"])

    # Load model and tokenizer from HF
    torch_dtype = torch.bfloat16 if general_config.get("torch_dtype", "bfloat16") == "bfloat16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch_dtype)

    # Create the LoRA config
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )
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

    # Create the DPO trainer
    trainer_config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=training_config.get("train_batch_size", 8),
        per_device_eval_batch_size=training_config.get("eval_batch_size", 8),
        gradient_accumulation_steps=training_config.get("grad_acc_steps", 2),
        num_train_epochs=training_config.get("epochs", 2),
        learning_rate=float(optimization_config.get("learning_rate", 5e-6)),
        # max_length=training_config.get("max_length", 1024),
        warmup_steps=lr_scheduling_config.get("num_warmup_steps", 10),
        lr_scheduler_type="cosine",
        logging_steps=training_config.get("logging_steps", 10),
        bf16=training_config.get("bf16", True),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        report_to=training_config.get("report_to", None),
        run_name=training_config.get("run_name", "DPO"),
        save_strategy=training_config.get("save_strategy", "no"),
        save_steps=training_config.get("save_steps", 500),
        max_steps=training_config.get("max_steps", -1),
    )

    # Adjust the dataset and training arguments for debug mode
    if args.debug:
        dataset = dataset.select(range(10))

    trainer = DPOTrainer(
        model=model,
        args=trainer_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    if trainer.is_world_process_zero() and training_config.get("report_to", None) == "wandb":
        wandb.init(
            project="ActiveUltraFeedback",
            entity="ActiveUF",
            name=training_config.get("run_name", "DPO"),
            config={
                "model": args.model_path,
                "dataset": args.dataset_path,
                "output_dir": args.output_dir,
                "seed": args.seed,
            },
        )

    trainer.train()

    # Save final model
    if trainer.is_world_process_zero():
        model = model.merge_and_unload()
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model and tokenizer saved to {args.output_dir}")

    # Export command line args for reproducibility
    with open(os.path.join(args.output_dir, "args.json"), "w") as f_out:
        json.dump(vars(args), f_out)