import numpy as np
import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class InformationDirectedSampling(BaseAcquisitionFunction):
    def __init__(self, beta=1.0, argmax_tol=1e-4, decision_buffer=0.0, prob_grid_size=100, rho2=1.0):
        super().__init__()
        self.beta = beta
        self.argmax_tol = argmax_tol
        self.decision_buffer = decision_buffer
        self.prob_grid_size = prob_grid_size
        self.rho2 = rho2
        self.rng = np.random.default_rng()
    def __call__(
        self,
        rewards: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> list[list[int, int]]:
        """
        IDS (Information Directed Sampling) for pairwise comparison selection.

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
        std_deviation = (upper_bounds - lower_bounds) / 2

        for p in range(rewards.shape[0]):
            r = rewards[p].cpu().numpy()
            s = std_deviation[p].cpu().numpy()

            posterior_mean, posterior_std = r, s
            n = posterior_mean.shape[0]

            # Step 1: UCB
            ucb = posterior_mean + self.beta * posterior_std
            cond_mask = np.logical_or(ucb > (0.5 - self.decision_buffer), np.eye(n, dtype=bool))
            candidate_mask = np.all(cond_mask, axis=1)

            if np.sum(candidate_mask) == 1:
                idx = np.argmax(candidate_mask)
                selected_pairs.append([idx, idx])
                continue

            # Step 2: Greedy arm selection
            null_idx = 0
            greedy_vals = np.where(candidate_mask, posterior_mean[:, null_idx], -np.inf)
            max_greedy = np.max(greedy_vals)
            mask_close = np.logical_and(
                np.abs(posterior_mean[:, null_idx] - max_greedy) < self.argmax_tol,
                candidate_mask
            )
            indices = np.flatnonzero(mask_close)
            if len(indices) == 0:
                greedy_idx = np.argmax(greedy_vals)
            else:
                greedy_idx = self.rng.choice(indices)

            if posterior_mean[greedy_idx, null_idx] < 0.5 and candidate_mask[null_idx]:
                greedy_idx = null_idx

            # Step 3: Suboptimality gap
            max_reward = np.max(np.where(candidate_mask, ucb[:, greedy_idx], -np.inf))
            suboptimality_gap = max_reward + posterior_mean[greedy_idx, :]  # shape: (n,)

            # Step 4: IDS computation
            prob_grid = np.linspace(0, 1, self.prob_grid_size + 1)[1:]  # skip 0
            prob_grid = prob_grid.reshape(1, -1)

            loss_sq = np.power(
                (1 - prob_grid) * max_reward + prob_grid * suboptimality_gap.reshape(-1, 1),
                2
            )
            info_gain = np.log(1 + posterior_std[greedy_idx, :].reshape(-1, 1) / self.rho2)
            ids = (loss_sq / prob_grid) * info_gain

            # Mask out greedy & invalid arms
            ids[greedy_idx, :] = np.inf
            for i in range(n):
                if not candidate_mask[i]:
                    ids[i, :] = np.inf

            # Step 5: Select minimum IDS point
            ids_min = np.min(ids)
            close_mask = np.abs(ids - ids_min) < self.argmax_tol
            idx_choices = np.argwhere(close_mask)

            if len(idx_choices) == 0:
                challenger_idx, prob_idx = np.unravel_index(np.argmin(ids), ids.shape)
            else:
                selected = self.rng.choice(len(idx_choices))
                challenger_idx, prob_idx = idx_choices[selected]

            p = prob_grid[0, prob_idx]
            if self.rng.uniform() < p:
                selected_pairs.append([greedy_idx, challenger_idx])
            else:
                selected_pairs.append([greedy_idx, greedy_idx])  # degenerate case

        return selected_pairs

