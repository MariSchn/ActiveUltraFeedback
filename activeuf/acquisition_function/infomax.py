import numpy as np
import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class InfoMax(BaseAcquisitionFunction):
    """
    Selects the pair of completions with the highest variance in their comparison.
    For each pair (i, j), computes the confidence gap when comparing them and 
    selects the pair with the maximum gap.
    """

    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__()
        self.beta = beta

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
        variances = (upper_bounds - lower_bounds)
        sorted_stds = torch.argsort(variances, descending=True, dim=-1)

        return sorted_stds[:, :2].tolist()
