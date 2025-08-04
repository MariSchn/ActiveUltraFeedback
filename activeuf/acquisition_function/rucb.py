import numpy as np
import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class RelativeUpperConfidenceBound(BaseAcquisitionFunction):
    def __init__(self, max_iterations: int = 10, beta: float = 1.0, argmax_tol: float = 1e-4, 
                 decision_buffer: float = 0.0):
        super().__init__()
        self.max_iterations = max_iterations
        self.beta = beta
        self.argmax_tol = argmax_tol
        self.rng: np.random.Generator = np.random.default_rng()
        self.decision_buffer = decision_buffer

    def __call__(
        self, 
        rewards: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> list[list[int, int]]:
        """
        RUCB (Relative Upper Confidence Bound) for pairwise comparison selection.

        Args:
            rewards: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the reward scores for each completion
            lower_bounds: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the lower bound / standard deviation of the reward for each completions
            upper_bounds: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the upper bound / standard deviation of the reward for each completions

                
        Returns:
            list of [i, j] per prompt where i is preferred, j is compared
        """
        selected_pairs = []
        n_prompts = rewards.shape[0]

        std_deviation = (upper_bounds - lower_bounds) / 2
        
        for p in range(n_prompts):
            r = rewards[p].cpu().numpy()
            s = std_deviation[p].cpu().numpy()

            posterior_mean, posterior_std = r, s
            n = posterior_mean.shape[0]

            ucb = posterior_mean + self.beta * posterior_std
            mask_diag = np.eye(n, dtype=bool)
            cond_mask = np.logical_or(ucb > (0.5 - self.decision_buffer), mask_diag)
            candidate_mask = np.all(cond_mask, axis=1)

            if np.sum(candidate_mask) == 1:
                idx = np.argmax(candidate_mask)
                selected_pairs.append([idx, idx])
                continue

            candidate_indices = np.flatnonzero(candidate_mask)
            next_j = self.rng.choice(candidate_indices)

            ucb_no_diag = np.copy(ucb)
            np.fill_diagonal(ucb_no_diag, -np.inf)
            ucb_j_col = ucb_no_diag[:, next_j]
            max_ucb = np.max(ucb_j_col)

            approx_max_mask = np.abs(ucb_j_col - max_ucb) < self.argmax_tol
            approx_max_indices = np.flatnonzero(approx_max_mask)

            if len(approx_max_indices) == 0:
                next_i = int(np.argmax(ucb_j_col))
            else:
                next_i = int(self.rng.choice(approx_max_indices))

            selected_pairs.append([next_i, next_j])

        return selected_pairs
