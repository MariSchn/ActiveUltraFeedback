from datasets import Dataset

from rewarduq.models.reward_head_ensemble import (
    RewardHeadEnsembleModel as ENNRewardModel,
    RewardHeadEnsembleModelConfig as ENNRewardModelConfig,
    RewardHeadEnsembleTrainer as ENNRewardModelTrainer,
    RewardHeadEnsembleTrainerConfig as ENNRewardModelTrainerConfig,
    RewardHeadEnsemblePipeline as ENNRewardModelPipeline,
)

from activeuf.utils import filter_dict

def custom_collate_fn(batch):
    out = {}
    for key in ["prompt_id", "prompt", "source", "completions", "row_id"]:
        out[key] = [x[key] for x in batch]
    return out

def compute_acquisition_function_KPIs(rewards, chosen_idxs, rejected_idxs):
    """
    Function to calculate acquisition function KPIs.
    rewards: Tensor of shape (n_samples, n_completions, 2) - rewards for each completion
    chosen_idxs: Tensor of shape (n_samples, 1) - indices of the chosen completions
    rejected_idxs: Tensor of shape (n_samples, 1) - indices of the rejected completions

    list of KPIs to calculate:
    - mean rewards and mean uncertainties of:
    --- all completions per sample
    --- chosen completions per sample
    --- rejected completions per sample
    --- same as above, but for the whole batch

    TODO: this assumes that the uncertainty bands are symmetric, but we agreed that this may not be the case

    TODO:
    track these KPIs for each model from the model pool separately.
    combined statistics for both chosen and rejected completions.
    """
    mean_rewards_per_sample = rewards.mean(dim=1)  # (n_samples, 2)
    mean_rewards_of_batch = mean_rewards_per_sample.mean(dim=0)  # (2,)

    chosen_rewards = rewards.gather(
        1, chosen_idxs.unsqueeze(-1).expand(-1, -1, rewards.size(-1))
    ).squeeze(1)
    rejected_rewards = rewards.gather(
        1, rejected_idxs.unsqueeze(-1).expand(-1, -1, rewards.size(-1))
    ).squeeze(1)

    mean_chosen_rewards = chosen_rewards.mean(dim=0)  # (2,)
    mean_rejected_rewards = rejected_rewards.mean(dim=0)  # (2,)

    # Add to KPIs
    kpis = {
        "mean_rewards_per_sample": mean_rewards_per_sample[:, 0].tolist(),
        "mean_rewards_per_batch": mean_rewards_of_batch[0].item(),
        "mean_uncertainty_per_sample": mean_rewards_per_sample[:, 1].tolist(),
        "mean_uncertainty_per_batch": mean_rewards_of_batch[1].item(),
        "mean_chosen_rewards_per_batch": mean_chosen_rewards[0].item(),
        "mean_chosen_uncertainty_per_batch": mean_chosen_rewards[1].item(),
        "mean_rejected_rewards_per_batch": mean_rejected_rewards[0].item(),
        "mean_rejected_uncertainty_per_batch": mean_rejected_rewards[1].item(),
        "chosen_rewards_per_sample": chosen_rewards[:, 0].tolist(),
        "chosen_uncertainty_per_sample": chosen_rewards[:, 1].tolist(),
        "rejected_rewards_per_sample": rejected_rewards[:, 0].tolist(),
        "rejected_uncertainty_per_sample": rejected_rewards[:, 1].tolist(),
    }
    return kpis

def init_model_tokenizer_trainer(args, batch_size):
    trainer_config = ENNRewardModelTrainerConfig(
        per_device_train_batch_size=batch_size,
        **filter_dict(args.reward_trainer_config[args.reward_model], ENNRewardModelTrainerConfig),
    )

    if args.previous_checkpoint_path:
        model = ENNRewardModel.from_pretrained(args.previous_checkpoint_path)
        tokenizer = model.tokenizer
    else:
        reward_model_pipeline = ENNRewardModelPipeline(
            ENNRewardModelConfig(**args.reward_model_config[args.reward_model]),
            trainer_config,
        )
        model = reward_model_pipeline.model
        tokenizer = model.tokenizer

    # TODO understand what's going on here
    # Initialize trainer with an empty Dataset having the required keys. So we have access to the uq_pipeline.trainer before entering the loop.
    trainer = ENNRewardModelTrainer(
        args=trainer_config,
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list([{
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
            "features_chosen": [],
            "features_rejected": [],
    }]))

    return model, tokenizer, trainer
