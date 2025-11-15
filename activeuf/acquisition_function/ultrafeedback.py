import numpy as np
import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class UltraFeedback(BaseAcquisitionFunction):
    """
    Selects indices in a way UltraFeedback does.
    It randomly selects 4 indices, finds the one with the highest reward,
    and then randomly selects another index from the remaining 3.
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
        Returns:
            list[list[int, int]]: The selected indices per prompt.
                The order for these is arbitrary and needs to be determined
                using an oracle.
        """
        n_prompts, n_completions = rewards.shape
        selected_indices = []
        for i in range(n_prompts):
            # Randomly choose 4 unique indices
            idxs = np.random.choice(n_completions, size=4, replace=False)
            # Find the index of the highest reward among the 4
            rewards_4 = rewards[i, idxs]
            max_idx_in_4 = torch.argmax(rewards_4).item()
            chosen_idx = idxs[max_idx_in_4]
            # Remove the chosen index and randomly select another from the rest
            remaining = np.delete(idxs, max_idx_in_4)
            second_idx = np.random.choice(remaining)
            selected_indices.append([chosen_idx, second_idx])
        return selected_indices
