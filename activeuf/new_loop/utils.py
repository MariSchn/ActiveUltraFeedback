from dataclasses import asdict
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

from activeuf.new_loop.arguments import ENNConfig

######################################
# OUTER/INNER LOOP UTILS
######################################


def custom_collate(batch):
    out = {}
    for key in [
        "prompt_id",
        "prompt",
        "source",
        "completions",
        "features",
    ]:
        out[key] = [x[key] for x in batch]
    return out


def custom_decollate(collated_batch):
    out = []
    for i in range(len(collated_batch["prompt_id"])):
        out.append(
            {
                "prompt_id": collated_batch["prompt_id"][i],
                "prompt": collated_batch["prompt"][i],
                "source": collated_batch["source"][i],
                "features": collated_batch["features"][i],
                "completions": collated_batch["completions"][i],
            }
        )
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


LOGS_CACHE = []


class WandbStepLoggerCallback(TrainerCallback):
    def __init__(self, accelerator):
        self.accelerator = accelerator

    def on_log(self, args, state, control, logs=None, **kwargs):
        global LOGS_CACHE
        if self.accelerator.is_main_process and logs is not None:
            for key in ["regularization_towards_initial_weights"]:
                logs[key] = getattr(args, key)

            # let current logs piggyback on the last entry in the cache
            if LOGS_CACHE:
                LOGS_CACHE[-1].update(logs)
            else:
                LOGS_CACHE.append(logs)

            for _logs in LOGS_CACHE:
                wandb.log(_logs)
            LOGS_CACHE = []


def init_model_trainer(
    reward_model_type: str, reward_args: ENNConfig | None, n_processes: int
):
    if reward_model_type == "none":
        model, trainer = None, None

    elif reward_model_type == "enn":
        trainer_config = ENNRewardModelTrainerConfig(
            per_device_train_batch_size=-(
                reward_args.effective_batch_size // -n_processes
            ),
            **asdict(reward_args.trainer),
        )

        if reward_args.previous_checkpoint_path:
            model = ENNRewardModel.from_pretrained(
                reward_args.previous_checkpoint_path,
            )
            tokenizer = model.tokenizer
        else:
            pipeline = ENNRewardModelPipeline(
                ENNRewardModelConfig(**asdict(reward_args.model)),
                trainer_config,
            )
            model = pipeline.model
            tokenizer = model.tokenizer

        # initialize trainer with an empty Dataset having the required keys. So we have access to the uq_pipeline.trainer before entering the loop.
        trainer = ENNRewardModelTrainer(
            args=trainer_config,
            model=model,
            processing_class=tokenizer,
            train_dataset=Dataset.from_list(
                [
                    {
                        "input_ids_chosen": [],
                        "attention_mask_chosen": [],
                        "input_ids_rejected": [],
                        "attention_mask_rejected": [],
                        "features_chosen": [],
                        "features_rejected": [],
                    }
                ]
            ),
        )
    else:
        raise NotImplementedError(f"{reward_model_type=} not implemented.")

    return model, trainer


def compute_rewards(samples, model, compute_reward_batch_size) -> torch.tensor:
    n_samples = len(samples)
    n_completions_per_sample = len(samples[0]["completions"])

    if model is None:
        return torch.zeros(
            (n_samples, n_completions_per_sample, 3),
            dtype=torch.float32,
        )

    def get_features_yielder():
        for sample in samples:
            for i in range(n_completions_per_sample):
                yield torch.tensor(sample["features"][i])

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
        features_mbatch = torch.stack(features_mbatch).to(model.device)

        with torch.no_grad():
            output = model(features=features_mbatch)

        rewards_batch.extend(output["rewards"].cpu())

    torch.cuda.empty_cache()
    rewards_batch = torch.stack(rewards_batch).view(n_samples, -1, 3)

    return rewards_batch


def get_acquired(samples, acquired_idxs):
    acquired = []
    for sample, (a, b) in zip(samples, acquired_idxs):
        completions = sample["completions"]

        acquired.append(
            {
                "prompt_id": sample["prompt_id"],
                "prompt": sample["prompt"],
                "source": sample["source"],
                "response_text_1": completions[a]["response_text"],
                "features_1": sample["features"][a],
                "1_model": completions[a]["model"],
                "1_score": completions[a]["overall_score"],
                "response_text_2": completions[b]["response_text"],
                "features_2": sample["features"][b],
                "2_model": completions[b]["model"],
                "2_score": completions[b]["overall_score"],
            }
        )
    return acquired


def compute_kpis(rewards, acquired_idxs) -> list[dict]:
    _rewards, _lower_bounds, _upper_bounds = rewards.unbind(-1)
    _uncertainty = (_upper_bounds - _lower_bounds) / 2  # TODO: why divide by 2?
    _chosen_idxs, _rejected_idxs = acquired_idxs.unbind(-1)

    index = torch.arange(_rewards.size(0))
    mean_rewards_per_sample = _rewards.mean(dim=1)
    mean_uncertainty_per_sample = _uncertainty.mean(dim=1)

    kpis = []
    for i in range(_rewards.size(0)):
        kpis.append(
            {
                "mean_rewards_per_sample": mean_rewards_per_sample[i].item(),
                "mean_uncertainty_per_sample": mean_uncertainty_per_sample[i].item(),
                "chosen_rewards_per_sample": _rewards[index, _chosen_idxs][i].item(),
                "rejected_rewards_per_sample": _rewards[index, _rejected_idxs][
                    i
                ].item(),
                "chosen_uncertainty_per_sample": _uncertainty[index, _chosen_idxs][
                    i
                ].item(),
                "rejected_uncertainty_per_sample": _uncertainty[index, _rejected_idxs][
                    i
                ].item(),
            }
        )
    return kpis


def restructure_sample(x: dict) -> dict:
    for key in ["chosen", "rejected"]:
        x[key] = [
            {"role": "user", "content": x["prompt"]},
            {"role": "assistant", "content": x[key]},
        ]
    return x


def get_new_regularization(
    n_done: int,
    n_total: int,
    decay_type: str,
    initial_value: float,
    exponential_decay_base: float = None,
) -> float:
    if decay_type == "linear":
        return initial_value * (1.0 - n_done / n_total)
    elif decay_type == "exponential":
        return initial_value * (exponential_decay_base**n_done)
    else:
        raise ValueError(f"{decay_type=} not supported")
