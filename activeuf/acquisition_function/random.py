import numpy as np
import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class RandomAcquisitionFunction(BaseAcquisitionFunction):
    """
    Randomly selects and returns two indices per prompt
    """
    def __call__(
        self,
        rewards: torch.Tensor,
        std_deviation: torch.Tensor,
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
        return np.random.randint(
            low=0,
            high=rewards.shape[1],
            size=(rewards.shape[0], 2),
        ).tolist()
