from datetime import datetime
from datasets import Dataset
from rewarduq.models.reward_head_ensemble import (
    RewardHeadEnsembleModel as ENNRewardModel,
    RewardHeadEnsembleModelConfig as ENNRewardModelConfig,
    RewardHeadEnsembleTrainer as ENNRewardModelTrainer,
    RewardHeadEnsembleTrainerConfig as ENNRewardModelTrainerConfig,
)

"""
Example run command:
accelerate launch --config_file=/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/activeuf/reward_model/multi_gpu.yaml \
    -m activeuf.OOM_demonstration
"""

enn_model_config = ENNRewardModelConfig(
            base_model_name_or_path="allenai/OLMo-2-1124-7B-SFT",
        )

enn_trainer_config = ENNRewardModelTrainerConfig(
            num_train_epochs=1,
            output_dir=f"trainer_output/{datetime.now().strftime("%Y%m%d-%H%M%S")}",
            save_strategy="no",
            per_device_train_batch_size=1,
            disable_tqdm=True,
            logging_strategy="steps",
            logging_steps=1,
            run_name=f"activeuf_{datetime.now().strftime("%Y%m%d-%H%M%S")}",
            lr_scheduler_type="constant",
            learning_rate=5e-6,
            report_to="none"
        )

dummy_data = [{
    'prompt': f'Prompt {i}',
    'prompt_id': f'id_{i}',
    'chosen': f'Chosen response {i}',
    'chosen_model': 'dummy-model',
    'chosen_score': 1,
    'input_ids_chosen': [101, 2000 + i, 102],  # Example token IDs
    'attention_mask_chosen': [1, 1, 1],
    'rejected': f'Rejected response {i}',
    'rejected_model': 'dummy-model',
    'rejected_score': 0,
    'input_ids_rejected': [101, 3000 + i, 102],  # Example token IDs
    'attention_mask_rejected': [1, 1, 1]
} for i in range(1)]


model = ENNRewardModel(enn_model_config)

trainer = ENNRewardModelTrainer(
        args=enn_trainer_config,
        model=model,
        processing_class=model.tokenizer,
        # compute_metrics=enn_compute_metrics,
        train_dataset=Dataset.from_list(dummy_data)
    )

print("Everything is set up, starting training...")
trainer.train()
print("Training completed successfully.")