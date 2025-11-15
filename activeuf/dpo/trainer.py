from dataclasses import dataclass, field
from typing import Literal, Union

import torch

from trl import DPOConfig, DPOTrainer


@dataclass
class NormedDPOConfig(DPOConfig):
    normalize_logps: bool = field(
        default=False,
        metadata={
            "help": "If `True`, all logps are normalized by the respective number of tokens."
        },
    )


class NormedDPOTrainer(DPOTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.normalize_logps = self.args.normalize_logps

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        model_output = self.concatenated_forward(model, batch)

        # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        chosen_logps = model_output["chosen_logps"]
        rejected_logps = model_output["rejected_logps"]
        if self.normalize_logps and self.loss_type == "sigmoid":
            chosen_lengths = batch["chosen_attention_mask"].sum(dim=1)
            rejected_lengths = batch["rejected_attention_mask"].sum(dim=1)

            chosen_logps = chosen_logps / chosen_lengths
            rejected_logps = rejected_logps / rejected_lengths
            ref_chosen_logps = ref_chosen_logps / chosen_lengths
            ref_rejected_logps = ref_rejected_logps / rejected_lengths

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = (
                losses + self.args.rpo_alpha * model_output["nll_loss"]
            )  # RPO loss from V3 of the paper

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        # sequence-wise proxy: log p_pi(x) - log p_ref(x); if normalized above, this is per-token KL
        kl_chosen = chosen_logps - ref_chosen_logps
        kl_rejected = rejected_logps - ref_rejected_logps
        # overall: average across both sets (simple concat then mean)
        kl_all = torch.cat([kl_chosen, kl_rejected], dim=0)

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = (
            self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        )
        metrics[f"{prefix}rewards/rejected"] = (
            self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        )
        metrics[f"{prefix}rewards/accuracies"] = (
            self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        )
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards)
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/chosen_normed"] = (
            self.accelerator.gather_for_metrics(chosen_logps).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected_normed"] = (
            self.accelerator.gather_for_metrics(rejected_logps).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"])
            .detach()
            .mean()
            .item()
        )
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"])
                .detach()
                .mean()
                .item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"])
                .detach()
                .mean()
                .item()
            )

        # --- Add KL metrics so W&B logs them ---
        metrics[f"{prefix}kl/chosen"] = (
            self.accelerator.gather_for_metrics(kl_chosen).mean().item()
        )
        metrics[f"{prefix}kl/rejected"] = (
            self.accelerator.gather_for_metrics(kl_rejected).mean().item()
        )
        metrics[f"{prefix}kl/overall"] = (
            self.accelerator.gather_for_metrics(kl_all).mean().item()
        )

        return losses.mean(), metrics
