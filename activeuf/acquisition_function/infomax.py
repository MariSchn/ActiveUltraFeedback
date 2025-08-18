import numpy as np
import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class InfoMax(BaseAcquisitionFunction):
    """
    Randomly selects and returns two indices per prompt
    """

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
            std_deviation: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the standard deviation of the reward for each completions
        Returns:
            list[list[int, int]]: The selected indices per prompt.
                The order for these is arbitrary and needs to be determined
                using an oracle.
        """
        std_deviation = (upper_bounds - lower_bounds) / 2

        sorted_stds = torch.argsort(std_deviation, descending=True, dim=-1)

        return sorted_stds[:, :2].tolist()
