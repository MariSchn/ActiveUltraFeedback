from datasets import Dataset
import torch
from transformers import TrainerCallback
import wandb

from rewarduq.models.reward_head_ensemble import (
    RewardHeadEnsembleModel as ENNRewardModel,
    RewardHeadEnsembleModelConfig as ENNRewardModelConfig,
    RewardHeadEnsembleTrainer as ENNRewardModelTrainer,
    RewardHeadEnsembleTrainerConfig as ENNRewardModelTrainerConfig,
    RewardHeadEnsemblePipeline as ENNRewardModelPipeline,
)

from activeuf.utils import filter_dict

def custom_collate(batch):
    out = {}
    for key in ["prompt_id", "prompt", "source", "completions", "row_id"]:
        out[key] = [x[key] for x in batch]
    return out

def custom_decollate(collated_batch):
    out = []
    for i in range(len(collated_batch["prompt_id"])):
        out.append({
            "prompt_id": collated_batch["prompt_id"][i],
            "prompt": collated_batch["prompt"][i],
            "source": collated_batch["source"][i],
            "row_id": collated_batch["row_id"][i],
            "completions": collated_batch["completions"][i],
        })
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

TRAINER_WANDB_STEP = 0
class WandbStepLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs, step=TRAINER_WANDB_STEP)

def init_reward_model_trainer(args: dict, n_processes: int): 
    if args.reward_model is None:
        reward_model, trainer, trainsize, initial_regularization = None, None, None, None

    elif args.reward_model == "enn":
        trainer_config_dict = args.reward_trainer_config[args.reward_model]
        trainer_config_dict["learning_rate"] = float(trainer_config_dict["learning_rate"])
        effective_batch_size = trainer_config_dict["effective_batch_size"]
        trainsize = effective_batch_size * trainer_config_dict["max_training_steps"]

        trainer_config = ENNRewardModelTrainerConfig(
            per_device_train_batch_size=-(effective_batch_size // -n_processes),
            **filter_dict(trainer_config_dict, ENNRewardModelTrainerConfig),
        )

        if args.previous_checkpoint_path:
            reward_model = ENNRewardModel.from_pretrained(args.previous_checkpoint_path)
            tokenizer = reward_model.tokenizer
        else:
            reward_model_pipeline = ENNRewardModelPipeline(
                ENNRewardModelConfig(**args.reward_model_config[args.reward_model]),
                trainer_config,
            )
            reward_model = reward_model_pipeline.model
            tokenizer = reward_model.tokenizer

        # initialize trainer with an empty Dataset having the required keys. So we have access to the uq_pipeline.trainer before entering the loop.
        trainer = ENNRewardModelTrainer(
            args=trainer_config,
            model=reward_model,
            processing_class=tokenizer,
            train_dataset=Dataset.from_list([{
                "input_ids_chosen": [],
                "attention_mask_chosen": [],
                "input_ids_rejected": [],
                "attention_mask_rejected": [],
                "features_chosen": [],
                "features_rejected": [],
        }]))

        initial_regularization = float(trainer.args.regularization_towards_initial_weights)
    else:
        raise NotImplementedError(f"Reward model {args.reward_model} not implemented.")

    return reward_model, trainer, trainsize, initial_regularization

def compute_rewards(samples, reward_model, compute_reward_batch_size) -> torch.tensor:
    n_samples = len(samples)
    n_completions_per_sample = len(samples[0]["completions"])
 
    if reward_model is None:
        return torch.zeros(
            (n_samples, n_completions_per_sample, 3),
            dtype=torch.float32,
        )

    def get_features_yielder():
        for sample in samples:
            for completion in sample["completions"]:
                yield torch.tensor(completion["features"])

    features_yielder = get_features_yielder()
    rewards_batch = []
    while True:
        features_mbatch = []
        for _ in range(compute_reward_batch_size):
            try:
                features_mbatch.append(next(features_yielder))
            except StopIteration:
                break
        if not features_mbatch:
            break
        features_mbatch = torch.stack(features_mbatch).to(reward_model.device)

        with torch.no_grad():
            output = reward_model(features=features_mbatch)

        rewards_batch.extend(output["rewards"].cpu())

    torch.cuda.empty_cache()
    rewards_batch = torch.stack(rewards_batch).view(n_samples, -1, 3)

    return rewards_batch

def get_acquired(samples, acquired_idxs):
    acquired = []
    for sample, (a, b) in zip(samples, acquired_idxs):
        completions = sample["completions"]

        acquired.append({
            "prompt_id": sample["prompt_id"],
            "prompt": sample["prompt"],
            "source": sample["source"],
            "row_id": sample["row_id"],

            "response_text_1": completions[a]["response_text"],
            "features_1": completions[a]["features"],
            "1_model": completions[a]["model"],
            "1_score": completions[a]["overall_score"],

            "response_text_2": completions[b]["response_text"],
            "features_2": completions[b]["features"],
            "2_model": completions[b]["model"],
            "2_score": completions[b]["overall_score"],
        })
    return acquired

def compute_kpi(rewards, acquired_idxs) -> tuple[list[dict], dict]:
    _rewards, _lower_bounds, _upper_bounds = rewards.unbind(-1)
    _uncertainty = (_upper_bounds - _lower_bounds) / 2  # TODO: why divide by 2?
    _chosen_idxs, _rejected_idxs = acquired_idxs.unbind(-1)
                                    
    mean_rewards_per_sample =_rewards.mean(dim=1)
    mean_uncertainty_per_sample = _uncertainty.mean(dim=1)

    index = torch.arange(_rewards.size(0))
    chosen_rewards_per_sample = _rewards[index, _chosen_idxs]
    rejected_rewards_per_sample = _rewards[index, _rejected_idxs]

    chosen_uncertainty_per_sample = _uncertainty[index, _chosen_idxs]
    rejected_uncertainty_per_sample = _uncertainty[index, _rejected_idxs]

    kpi_samplewise = []
    for i in range(_rewards.size(0)):
        kpi_samplewise.append({
            "mean_rewards_per_sample": mean_rewards_per_sample[i].item(),
            "mean_uncertainty_per_sample": mean_uncertainty_per_sample[i].item(),

            "chosen_rewards_per_sample": chosen_rewards_per_sample[i].item(),
            "rejected_rewards_per_sample": rejected_rewards_per_sample[i].item(),

            "chosen_uncertainty_per_sample": chosen_uncertainty_per_sample[i].item(),
            "rejected_uncertainty_per_sample": rejected_uncertainty_per_sample[i].item()
        })
    
    kpi_minibatchwise = {
        "mean_rewards_per_minibatch": mean_rewards_per_sample.mean().item(),
        "mean_uncertainty_per_minibatch": mean_uncertainty_per_sample.mean().item(),

        "mean_chosen_rewards_per_minibatch": chosen_rewards_per_sample.mean().item(),
        "mean_rejected_rewards_per_minibatch": rejected_rewards_per_sample.mean().item(),

        "mean_chosen_uncertainty_per_minibatch": chosen_uncertainty_per_sample.mean().item(),
        "mean_rejected_uncertainty_per_minibatch": rejected_uncertainty_per_sample.mean().item(),
    }

    return kpi_samplewise, kpi_minibatchwise

def restructure_sample(x: dict) -> dict:
    for key in ["chosen", "rejected"]:
        x[key] = [
            {"role": "user", "content": x["prompt"]},
            {"role": "assistant", "content": x[key]},
        ]
    return x

def update_regularization(
    initial_lambda_regularization: float,
    outer_batch_idx: int,
    outer_loop_batch_size: int,
    trainer_config_dict: dict,
) -> float:
    decay_type = trainer_config_dict.get("regularization_weight_decay_type")
    if decay_type == "linear":
        return (
            initial_lambda_regularization * outer_loop_batch_size
            / ((outer_batch_idx + 1) * outer_loop_batch_size)
        )
    elif decay_type == "exponential":
        return initial_lambda_regularization * (
            trainer_config_dict.get("exponential_decay_base") ** outer_batch_idx
        )
    else:
        return initial_lambda_regularization