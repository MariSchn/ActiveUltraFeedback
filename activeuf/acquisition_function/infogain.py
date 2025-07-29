import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction

class InfoGain(BaseAcquisitionFunction):
    """
    Selects the first response via Thompson Sampling and the second
    by identifying whichever leads to the highest information gain.
    Based on the implementation in https://github.com/sail-sg/oat.
    """
    def __call__(
        self,
        rewards: torch.Tensor,
        upper_bounds: torch.Tensor,
        lower_bounds: torch.Tensor,
    ) -> list[list[int, int]]:
        """
        Args:
            rewards: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the reward scores for each completion
            upper_bounds: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the upper bound of the reward for each completions
            lower_bounds: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the lower_bound of the reward for each completions
        Returns:
            list[list[int, int]]: The selected indices per prompt.
                The order for these is arbitrary and needs to be determined
                using an oracle.
        """
        # sample first action as action with highest reward
        first_idxs = rewards.argmax(axis=1)

        # determine confidence bounds for whether first action is better than each possible action
        upper_confidence_bounds = torch.sigmoid(
            upper_bounds.gather(axis=1, index=first_idxs.unsqueeze(1)) - lower_bounds)
        lower_confidence_bounds = torch.sigmoid(
            lower_bounds.gather(axis=1, index=first_idxs.unsqueeze(1)) - upper_bounds)
        confidence_gap_sizes = upper_confidence_bounds - lower_confidence_bounds

        # set gap size for the first action to very negative number (so that the second action does not collide with the first action)
        M, _ = rewards.shape
        confidence_gap_sizes[torch.arange(M), first_idxs] = -1e7

        #sample second action as action with largest confidence gap
        second_idxs = confidence_gap_sizes.argmax(axis=1)

        return list(zip(first_idxs.tolist(), second_idxs.tolist()))