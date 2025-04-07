from activeuf.configs import *
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset

#TODO: Refactoring opportunities
chat_templates = {
   "microsoft/deberta-v3-base": """
                                    {% for message in messages %}
                                    {{ message['role'] }}: {{ message['content'] }}
                                    {% endfor %}
                                """
}

#TODO:
# class for training reward models, which will take base models, datasets, and configurations

def train_and_save_model(output_dir="TrainedRewardModel", base_model="microsoft/deberta-v3-base"):

    # 1. Load dataset
    dataset = load_dataset("trl-lib/ultrafeedback_binarized")

    model_name = base_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.chat_template is None:
        print("Chat template not present.\nManually assigning a chat template\n")
        tokenizer.chat_template = chat_templates[model_name]
    # for adding a head on top of the LLM logits
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)


    # 3. Configure LoRA - Simplified configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_proj", "key_proj", "value_proj"], #can be experimented
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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

    print(model)
    
    print(f"Model and tokenizer saved to {output_dir}")

train_and_save_model()
