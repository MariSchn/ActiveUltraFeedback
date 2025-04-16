from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset
from activeuf.configs import MODEL_MAP, MODEL2CHAT_TEMPLATE

#TODO:
# general class for training reward models, which will take base models, datasets, and configurations

def train_and_save_model(output_dir="TrainedRewardModel", base_model="deberta-v3-base"):

    # 1. Load dataset
    dataset = load_dataset("trl-lib/ultrafeedback_binarized")

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
        r=16,
        lora_alpha=32,
        #target_modules=["query_proj", "key_proj", "value_proj"], #can be experimented
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
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
        per_device_train_batch_size=2, #Due to GPU constraints.
        num_train_epochs=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=500,
        remove_unused_columns=False,
        optim="adamw_torch",
        max_length=512,
        report_to="none",
        max_steps=3,
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

train_and_save_model(output_dir="reward_model", base_model="deberta-v3-base")
