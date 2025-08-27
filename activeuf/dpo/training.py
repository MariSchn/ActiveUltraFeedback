import argparse
import os
import shutil
import yaml

from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

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

    # load dataset, remove unnecessary columns
    print(f"Loading dataset from {config["train_path"]}...\n")
    dataset = load_from_disk(config["train_path"])
    if all(_ in dataset.column_names for _ in ["messages", "prompt", "chosen", "rejected"]):
        print("Warning: Dataset contains 'messages' column along with 'prompt', 'chosen', 'rejected'. Removing 'messages' column to avoid conflict.")
        dataset = dataset.remove_columns(["messages"])

    # limit dataset if in debug mode
    if config.get("debug"):
        dataset = dataset.select(range(5000))

    # load tokenizer and model from HF
    model_path = config["base_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch_dtype)

    # create lora version of model
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
            print(f"Target module patterns for {model_path} must be implemented manually")
            raise
        
    # create DPO trainer
    trainer_config = DPOConfig(run_name=run_name, output_dir=output_dir, **training_config)
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