def custom_collate_fn(batch):
    return {
        "prompt_id": [x["prompt_id"] for x in batch],
        "prompt": [x["prompt"] for x in batch],
        "source": [x["source"] for x in batch],
        "completions": [x["completions"] for x in batch],
        "row_id": [x["row_id"] for x in batch],
    }

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