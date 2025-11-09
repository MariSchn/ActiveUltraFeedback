import numpy as np
import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class InfoMax(BaseAcquisitionFunction):
    """
    Selects the pair of completions with the highest variance in their comparison.
    For each pair (i, j), computes the confidence gap when comparing them and 
    selects the pair with the maximum gap.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def __call__(
        self,
        rewards: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> list[list[int, int]]:
        """
        Args:
            rewards: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the reward scores for each completion
            lower_bounds: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the lower_bound of the reward for each completions
            upper_bounds: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the upper bound of the reward for each completions
        Returns:
            list[list[int, int]]: The selected indices per prompt.
                The order for these is arbitrary and needs to be determined
                using an oracle.
        """
        n_prompts, n_completions = rewards.shape
        
        # Compute confidence bounds for all pairs (i, j) and compute the confidence gap
        upper_confidence_bounds = torch.sigmoid(upper_bounds.unsqueeze(2) - upper_bounds.unsqueeze(1))
        lower_confidence_bounds = torch.sigmoid(lower_bounds.unsqueeze(2) - lower_bounds.unsqueeze(1))
        
        confidence_gaps = upper_confidence_bounds - lower_confidence_bounds

        # Mask the diagonal to prevent choosing the same completion for both i and j
        diag_mask = torch.eye(n_completions, device=confidence_gaps.device, dtype=torch.bool).unsqueeze(0)
        confidence_gaps.masked_fill_(diag_mask, -torch.inf)

        # Find the pair with the largest confidence gap (flatten the n_completions x n_completions matrix, find argmax and convert back to (i, j) pairs)
        confidence_gaps_flattened = confidence_gaps.view(n_prompts, -1)
        max_gap_flat_idx = confidence_gaps_flattened.argmax(dim=1)
        
        first_idxs = max_gap_flat_idx // n_completions
        second_idxs = max_gap_flat_idx % n_completions
        
        return list(zip(first_idxs.tolist(), second_idxs.tolist()))
