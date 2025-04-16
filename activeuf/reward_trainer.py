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
                                """,
    "facebook/opt-125m":        """
                                    {% for message in messages %}
                                    {{ message['role'] }}: {{ message['content'] }}
                                    {% endfor %}
                                """,                                
}
chat_template = """
                    {% for message in messages %}
                    {{ message['role'] }}: {{ message['content'] }}
                    {% endfor %}
                """

#TODO:
# class for training reward models, which will take base models, datasets, and configurations

def train_and_save_model(output_dir="TrainedRewardModel", base_model="microsoft/deberta-v3-base", false_training=False):

    # 1. Load dataset
    dataset = load_dataset("trl-lib/ultrafeedback_binarized")

    model_name = base_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
 
    # for adding a head on top of the LLM logits
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    #It turns out the RewardTrainer internally uses chat templates to format the training dataset
    #for example, when it takes the "chosen" response, it should format the prompt and completion.
    #some datasets don't come already formatted...
    if tokenizer.chat_template is None:
        print("Chat template not present.\nManually assigning a chat template\n")
        tokenizer.chat_template = chat_template # chat_templates[model_name]
        
    if tokenizer.pad_token is None:
        print("No pad_token present. Assigning tokenizer.pad_token = tokenizer.eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        
    # tokenizer.model_max_length = 512  # Match your BERT config
    # tokenizer.truncation_side = "right"  # Truncate from the end (or "left")
    # model.config.max_model_input_sizes = 512
    
    print(model)
    print(model.config)

    # 3. Configure LoRA - Simplified configuration
    
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
    
    model.print_trainable_parameters()
    tokenizer.model_max_length = 512
    tokenizer.truncation_side = "right"

    # Enable truncation by default (via config)
    tokenizer.init_kwargs["truncation"] = True
    tokenizer.init_kwargs["max_length"] = 512

    
    if false_training:
        print("Not training the model, just savining the weights.")
        model = model.merge_and_unload()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return 

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

#train_and_save_model(output_dir="reward7", base_model="allenai/longformer-base-4096", false_training=True)
# train_and_save_model(output_dir="reward8", base_model="EleutherAI/gpt-neo-125M", false_training=True)
train_and_save_model(output_dir="reward9", base_model="microsoft/deberta-v3-base", false_training=True)
