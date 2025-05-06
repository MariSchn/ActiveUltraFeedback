import os
import argparse
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from datasets import load_dataset
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, get_peft_model


def train_reward_model(config, args):
    output_dir = args.output_dir
    general_config = config.get("general", {})
    training_config = config.get("training", {})
    optimization_config = config.get("optimization", {})
    lr_scheduling_config = config.get("lr_scheduling", {})
    lora_config = config.get("lora", {})

    base_model = general_config.get("base_model", "meta-llama/Llama-3.2-1B-Instruct")
    dataset_name = general_config.get("dataset_name", "trl-lib/ultrafeedback_binarized")
    seed = general_config.get("seed", 42)
    set_seed(seed)
    torch_dtype = torch.bfloat16 if general_config.get("torch_dtype", "bfloat16") == "bfloat16" else torch.float32

    # Load dataset
    dataset = load_dataset(dataset_name)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        torch_dtype=torch_dtype,
    )

    if tokenizer.pad_token is None:
        print("No pad_token present. Assigning tokenizer.pad_token = tokenizer.eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "SEQ_CLS"),
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

    trainer_config = RewardConfig(
        output_dir=output_dir,
        per_device_train_batch_size=training_config.get("train_batch_size", 8),
        per_device_eval_batch_size=training_config.get("eval_batch_size", 8),
        gradient_accumulation_steps=training_config.get("grad_acc_steps", 2),
        num_train_epochs=training_config.get("epochs", 1),
        learning_rate=float(optimization_config.get("lr", 2e-5)),
        max_length=training_config.get("max_length", 1024),
        warmup_steps=lr_scheduling_config.get("num_warmup_steps", 10),
        logging_steps=training_config.get("logging_steps", 10),
        bf16=training_config.get("bf16", True),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        report_to=training_config.get("report_to", None),
        save_strategy=training_config.get("save_strategy", "no"),
        save_steps=training_config.get("save_steps", 500),
        max_steps=training_config.get("max_steps", -1),
    )

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=trainer_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"], # Currenty not used
    )

    trainer.train()


    # Save final model
    if trainer.is_world_process_zero():
        model = model.merge_and_unload()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")


#These utility functions could be moved to a separate file for better organization
def parse_arguments():
    parser = argparse.ArgumentParser(description="Reward model training using RewardTrainer.")
    parser.add_argument("--reward_config", required=True, help="Path to the YAML reward training config.")
    parser.add_argument("--output_dir", required=True, help="Directory to save trained model.")
    return parser.parse_args()


def load_config(config_path):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {e}\nUsing default configuration for training.\n")
        return {}


def main():
    args = parse_arguments()
    config = load_config(args.reward_config)
    train_reward_model(config, args)


if __name__ == "__main__":
    main()
