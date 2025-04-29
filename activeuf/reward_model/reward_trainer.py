import os
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
import argparse
import yaml

def train_and_save_model(config, args):
    output_dir = args.output_dir
    general_config = config.get("general", {})
    dataset_name = general_config.get("dataset_name", "trl-lib/ultrafeedback_binarized")
    base_model = general_config.get("base_model", "meta-llama/Llama-3.2-1B-Instruct")
    random_seed = general_config.get("seed", 42)
    lora_config = config.get("lora", {})
    training_config = config.get("training", {})
    optimization_config = config.get("optimization", {})
    lr_scheduling_config = config.get("lr_scheduling", {})
    
    accelerator = Accelerator()
    set_seed(random_seed)

    # Load dataset
    dataset = load_dataset(dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        torch_dtype=torch.bfloat16
    )

    if tokenizer.pad_token is None:
        accelerator.print("No pad_token present. Assigning tokenizer.pad_token = tokenizer.eos_token\n")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "SEQ_CLS")
    )

    try:
        model = get_peft_model(model, peft_config)
    except Exception as e:
        accelerator.print(f"Exception when applying LoRA: {e}")
        try:
            target_modules = []
            patterns = lora_config.get("target_module_patterns", ["query", "key", "value"])
            for name, _ in model.named_modules():
                if any(p in name for p in patterns):
                    target_modules.append(name)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)
        except Exception as e:
            accelerator.print(f"Exception when trying to identify target module parameters: {e}")
            exit(-1)
        
    # Preprocessing: get prompt + chosen + rejected
    def preprocess_function(examples):
        prompts = []
        chosen_responses = []
        rejected_responses = []

        for chosen_conv, rejected_conv in zip(examples["chosen"], examples["rejected"]):
            # Extract user prompt (from the chosen conversation)
            user_prompt = next((msg["content"] for msg in chosen_conv if msg["role"] == "user"), "")
            prompts.append(user_prompt)

            # Extract assistant responses
            chosen_reply = next((msg["content"] for msg in chosen_conv if msg["role"] == "assistant"), "")
            rejected_reply = next((msg["content"] for msg in rejected_conv if msg["role"] == "assistant"), "")

            chosen_responses.append(chosen_reply)
            rejected_responses.append(rejected_reply)

        return {
            "prompt": prompts,
            "chosen": chosen_responses,
            "rejected": rejected_responses,
        }

    with accelerator.main_process_first():
        reward_datasets = dataset.map(preprocess_function, batched=True)
        # testing training speed on different GPU sizes for specific size datasets.
        # reward_datasets["train"] = reward_datasets["train"].select(range(8000))
        # print(f"\n\n Length is: {len(reward_datasets["train"])}\n\n\n")

    train_dataset = reward_datasets["train"]
    eval_dataset = reward_datasets["test"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=training_config.get("train_batch_size", 8))
    eval_dataloader = DataLoader(eval_dataset, batch_size=training_config.get("eval_batch_size", 8))

    optimizer = AdamW(model.parameters(), lr=float(optimization_config.get("lr", 2e-5)))

    num_training_steps = training_config.get("max_steps", 100) # Limit total steps
    # total elements considered per step:     num_processes * BATCH_SIZE_PER_GPU  or  NUM_GPUS * BATCH_SIZE_PER_GPU
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_scheduling_config.get("num_warmup_steps", 10),
        num_training_steps=num_training_steps,
    )

    # Prepare all via Accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    model.train()
    completed_steps = 0
    gradient_accumulation_steps = training_config.get("grad_acc_steps", 2)
    logging_steps = training_config.get("logging_steps", 10)
    for epoch in range(training_config.get("epochs", 1)):
        for step, batch in enumerate(train_dataloader):
            # Tokenize chosen and rejected responses
            chosen_inputs = tokenizer(
                batch["prompt"],
                batch["chosen"],
                padding=training_config.get("padding", "max_length"),
                truncation=training_config.get("truncation", True),
                max_length=training_config.get("max_length", 1024),
                return_tensors=training_config.get("return_tensors", "pt")
            ).to(accelerator.device)

            rejected_inputs = tokenizer(
                batch["prompt"],
                batch["rejected"],
                padding=training_config.get("padding", "max_length"),
                truncation=training_config.get("truncation", True),
                max_length=training_config.get("max_length", 1024),
                return_tensors=training_config.get("return_tensors", "pt")
            ).to(accelerator.device)

            # Get model outputs
            chosen_rewards = model(**chosen_inputs).logits.squeeze(-1)
            rejected_rewards = model(**rejected_inputs).logits.squeeze(-1)

            # Compute reward loss
            loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
            loss = loss / gradient_accumulation_steps

            accelerator.backward(loss)

            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and step % logging_steps == 0:
                accelerator.print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

            completed_steps += 1
            if completed_steps >= num_training_steps:
                break

        if completed_steps >= num_training_steps:
            break

    # Merge LoRA into base model
    model = model.module
    model = model.merge_and_unload()

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        accelerator.print(f"Model and tokenizer saved to {output_dir}")

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple argument parser for reward trainer.")
    parser.add_argument("--reward_config", required=True, help="Path to the reward training configuration YAML file.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    args = parser.parse_args()
    
    try:
        # Attempt to load the reward configuration file
        with open(args.reward_config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: The specified reward configuration file '{args.reward_config}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse the reward configuration file '{args.reward_config}'.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading the reward configuration file: {e}")
    finally:
        print("Continuing with default parameters")
        
    train_and_save_model(config=config, args=args)
