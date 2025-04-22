from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset
from activeuf.configs import MODEL_MAP, MODEL2CHAT_TEMPLATE
import argparse
import yaml

#TODO:
# general class for training reward models, which will take base models, datasets, and configurations

#TODO: handle scientific notation with argumentparser, so it doesn't convert 1e-5 into a string.

def train_and_save_model(config):
    output_dir = config.get("output_dir", "TrainedRewardModel")
    base_model = config.get("base_model", "deberta-v3-base")
    dataset_name = config.get("dataset_name", "trl-lib/ultrafeedback_binarized")
    lora_config = config.get("lora", {})
    training_config = config.get("training", {})
    
    # 1. Load dataset
    dataset = load_dataset(dataset_name)

    model_name = MODEL_MAP[base_model]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
 
    # for adding a head on top of the LLM logits
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    if tokenizer.chat_template is None:
        print("Chat template not present.\nManually assigning a chat template\n")
        tokenizer.chat_template = MODEL2CHAT_TEMPLATE[base_model]
        
    if tokenizer.pad_token is None:
        print("No pad_token present. Assigning tokenizer.pad_token = tokenizer.eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        
    # 3. Configure LoRA - Simplified configuration    
    # TODO: model architecture should be known, otherwise, we will have to manually assign target_modules
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "SEQ_CLS")
    )

    try:
        model = get_peft_model(model, peft_config)
    except: # If we are asked to give target_modules explicitly.
        print("Unable to train the model with no target_modules defined.\nManually searching for module names including: [key, value, query]\n")
        target_modules = []
        patterns=["query", "key", "value"]
        for name, module in model.named_modules():
            if any(pattern in name for pattern in patterns):
                target_modules.append(name)
        peft_config.target_modules = target_modules
        model = get_peft_model(model, peft_config)
    
    # 4. Training
    training_args = RewardConfig(
        per_device_train_batch_size=training_config.get("batch_size", 1),
        num_train_epochs=training_config.get("epochs", 2),
        gradient_accumulation_steps=training_config.get("grad_acc_steps", 2),
        learning_rate=float(training_config.get("learning_rate", 2e-5)),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 500),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        optim=training_config.get("optimizer", "adamw_torch"),
        max_length=training_config.get("max_length", 1024),
        report_to=training_config.get("report_to", "none"),
        max_steps=training_config.get("max_steps", 200),
        output_dir=output_dir
    )
    
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
    )

    trainer.train()

    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train reward model using reward config YAML.")
    parser.add_argument("--reward_config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    with open(args.reward_config, "r") as f:
        config = yaml.safe_load(f)
    
    train_and_save_model(config)
