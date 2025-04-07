def train_and_save_model(output_dir="final_reward_model"):

    from peft import LoraConfig
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
    from trl import RewardTrainer, RewardConfig
    from datasets import load_dataset
    from utils import load_model

    # 1. Load dataset
    dataset = load_dataset("trl-lib/ultrafeedback_binarized")

    # 2. Initialize model and tokenizer
    model_name = "gpt2"
    model = load_model(model_name=model_name)
    tokenizer = model.get_tokenizer()

    # 3. Configure LoRA - Simplified configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_proj"],  # Only target attention projection layers
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )

    # 4. Training
    training_args = RewardConfig(
        per_device_train_batch_size=8,
        num_train_epochs=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=500,
        remove_unused_columns=False,
        optim="adamw_torch",
        max_length=512,
        report_to="none",
        max_steps=1,
    )

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
    )

    # 5. Train
    trainer.train()
    trainer.save_model(output_dir)
    return model, tokenizer
