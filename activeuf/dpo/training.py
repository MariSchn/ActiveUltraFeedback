import argparse
import os
import wandb
import math
import shutil

from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

from accelerate import Accelerator

from activeuf.configs import *
from activeuf.utils import *

"""
This script is designed to train a (SFT) model using direct preference optimization (DPO).
To train the DPO model a (previously SFT trained) model is required and a binarized dataset.
The binarized dataset is a dataset that has been preprocessed to contain pairs of examples, where one example is preferred over the other.

Example run command:
    accelerate launch \
        --config_file $SCRATCH/ActiveUltraFeedback/activeuf/dpo/accelerate_config.yml -m activeuf.dpo.training \
        --config_path $SCRATCH/ActiveUltraFeedback/activeuf/dpo/training.yaml \
        --debug
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--config_path", type=str,
                        help="The path to the YAML training config.")

    parser.add_argument("--debug", action="store_true",
                        help="Whether to run in debug mode (just one train step on a small subset of the dataset)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    setup(login_to_wandb=True)

    # Parse config file
    config = load_config(args.config_path) if args.config_path else {}
    general_config = config.get("general", {})
    training_config = config.get("training", {})
    optimization_config = config.get("optimization", {})
    lr_scheduling_config = config.get("lr_scheduling", {})
    lora_config = config.get("lora", {})

    # Set random seed
    if isinstance(general_config.get("seed"), int):
        set_seed(general_config["seed"])

    # setup accelerate
    accelerator = Accelerator()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load the dataset
    print(f"Loading dataset from {args.train_path}...")
    dataset = load_from_disk(args.train_path)

    # Adjust the dataset and training arguments for debug mode
    if args.debug:
        dataset = dataset.select(range(100))

    # sanitise dataset columns
    if "messages" in dataset.column_names and all(k in dataset.column_names for k in ["prompt", "chosen", "rejected"]):
        print("Warning: Dataset contains 'messages' column along with 'prompt', 'chosen', 'rejected'. Removing 'messages' column to avoid conflict.")
        dataset = dataset.remove_columns(["messages"])

    # Load model and tokenizer from HF
    if general_config.get("torch_dtype", "bfloat16") == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    base_model = general_config.get("base_model", "allenai/Llama-3.1-Tulu-3-8B-SFT")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch_dtype)

    # Create the LoRA config
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )

    # Prepare PEFT model
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

    # Prepare output dir and run name to be basically the same
    timestamp = get_timestamp()
    dataset_name = os.path.basename(args.train_path).replace("-", "_")
    base_model_name = os.path.basename(base_model).replace("-", "_")
    output_dir = os.path.join(
        general_config.get("output_dir", "models/dpo"), 
        dataset_name, base_model_name, timestamp,
    )
    run_name = f"{dataset_name}-{base_model_name}-{timestamp}"

    # Prepare trainer
    trainer_config = DPOConfig(
        report_to=general_config.get("report_to", "wandb"),
        run_name=run_name,
        output_dir=output_dir,

        num_train_epochs=training_config.get("epochs", 2),
        per_device_train_batch_size=math.ceil(
            training_config.get("train_batch_size", 32) / accelerator.num_processes),
        gradient_accumulation_steps=training_config.get("grad_acc_steps", 2),
        max_length=training_config.get("max_length", 4096),
        bf16=training_config.get("bf16", True),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        save_strategy=training_config.get("save_strategy", "no"),

        save_steps=training_config.get("save_steps", 500),
        logging_steps=training_config.get("logging_steps", 10),
        max_steps=training_config.get("max_steps", -1),

        learning_rate=float(optimization_config.get("learning_rate", 5e-6)),

        lr_scheduler_type=lr_scheduling_config.get("lr_scheduler_type", "cosine"),
        warmup_steps=lr_scheduling_config.get("num_warmup_steps", 150),
    )
    trainer = DPOTrainer(
        model=model,
        args=trainer_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    # Save model, tokenizer, and configs
    if trainer.is_world_process_zero():
        model = model.merge_and_unload()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        shutil.copy(
            args.config_path, 
            os.path.join(output_dir, os.path.basename(args.config_path))
        )
        print(f"Model, tokenizer, and configs saved to {output_dir}")