import numpy as np
import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class InfoGain(BaseAcquisitionFunction):
    """
    Selects the first response via Thompson Sampling and the second
    by identifying whichever leads to the highest information gain.
    Based on the implementation in https://github.com/sail-sg/oat.
    """

    def __init__(self, beta: int = 1, **kwargs):
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
        # sample first action as action with highest reward
        std_devs = (upper_bounds - lower_bounds) / 2
        first_idxs = torch.tensor(
            [
                self.dts_optimize(_rewards, _std_devs)
                for _rewards, _std_devs in zip(rewards, std_devs)
            ]
        )

        # determine confidence bounds for whether first action is better than each possible action
        upper_confidence_bounds = torch.sigmoid(
            upper_bounds.gather(dim=1, index=first_idxs.unsqueeze(1)) - lower_bounds
        )
        lower_confidence_bounds = torch.sigmoid(
            lower_bounds.gather(dim=1, index=first_idxs.unsqueeze(1)) - upper_bounds
        )
        confidence_gap_sizes = upper_confidence_bounds - lower_confidence_bounds

        # set gap size for the first action to very negative number (so that the second action does not collide with the first action)
        # original paper resamples to avoid collision until max iteration is hit, rather than mask it out. we choose to mask instead because it's faster, and we are not accepting collisions
        n_prompts, _ = rewards.shape
        confidence_gap_sizes[torch.arange(n_prompts), first_idxs] = -1e7

        # sample second action as action with largest confidence gap
        second_idxs = confidence_gap_sizes.argmax(dim=1)

        return list(zip(first_idxs.tolist(), second_idxs.tolist()))

    def dts_optimize(self, _rewards, _std_devs):
        """
        Args:
            _rewards: tensor of shape (n_completions_per_prompt,)
            _std_devs: tensor of shape (n_completions_per_prompt,)
        """
        r_epistemic_index = []
        for j in range(len(_rewards)):
            z = np.random.uniform(-1, 1)

            r_x_y_epistemic_index = _rewards[j] + self.beta * z * _std_devs[j]
            r_epistemic_index.append(r_x_y_epistemic_index)
        return np.argmax([idx.cpu() for idx in r_epistemic_index])
