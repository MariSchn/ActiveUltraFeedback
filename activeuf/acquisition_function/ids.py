import numpy as np
import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class InformationDirectedSampling(BaseAcquisitionFunction):
    def __init__(self, beta=1.0, argmax_tol=1e-4, decision_buffer=0.0,
                 prob_grid_size=100, rho2=1.0):
        super().__init__()
        self.beta = beta
        self.argmax_tol = argmax_tol
        self.decision_buffer = decision_buffer
        self.prob_grid_size = prob_grid_size
        self.rho2 = rho2
        self.rng = np.random.default_rng()

    def __call__(self, rewards: torch.Tensor,
                 lower_bounds: torch.Tensor,
                 upper_bounds: torch.Tensor) -> list[list[int, int]]:
        """
        IDS (Information Directed Sampling) in vector form (per-arm, not pairwise).

        Args:
            rewards: tensor (n_prompts, n_completions_per_prompt)
            lower_bounds: tensor (n_prompts, n_completions_per_prompt)
            upper_bounds: tensor (n_prompts, n_completions_per_prompt)
        """
        selected_pairs = []
        std_deviation = (upper_bounds - lower_bounds) / 2  # (n_prompts, n_completions)

        for p in range(rewards.shape[0]):
            r = rewards[p].cpu().numpy()       # shape (n,)
            s = std_deviation[p].cpu().numpy() # shape (n,)

            posterior_mean, posterior_std = r, s
            n = posterior_mean.shape[0]

            # Step 1: Candidate set based on UCB
            ucb = posterior_mean + self.beta * posterior_std
            candidate_mask = ucb > (0.5 - self.decision_buffer)

            if np.sum(candidate_mask) == 1:
                idx = int(np.argmax(candidate_mask))
                selected_pairs.append([idx, idx])
                continue

            # Step 2: Greedy arm selection (null_idx = 0 for reference)
            greedy_vals = np.where(candidate_mask, posterior_mean, -np.inf)
            max_greedy = np.max(greedy_vals)
            mask_close = np.logical_and(
                np.abs(posterior_mean - max_greedy) < self.argmax_tol,
                candidate_mask
            )
            indices = np.flatnonzero(mask_close)
            if len(indices) == 0:
                greedy_idx = int(np.argmax(greedy_vals))
            else:
                greedy_idx = int(self.rng.choice(indices))

            if posterior_mean[greedy_idx] < 0.5 and candidate_mask[0]:
                greedy_idx = 0

            # Step 3: Suboptimality gap
            max_reward = np.max(np.where(candidate_mask, ucb, -np.inf))
            suboptimality_gap = max_reward + posterior_mean  # shape (n,)

            # Step 4: IDS computation over probability grid
            prob_grid = np.linspace(0, 1, self.prob_grid_size + 1)[1:]  # skip p=0
            prob_grid = prob_grid.reshape(1, -1)

            loss_sq = np.power(
                (1 - prob_grid) * max_reward + prob_grid * suboptimality_gap.reshape(-1, 1),
                2
            )
            info_gain = np.log(1 + posterior_std.reshape(-1, 1) / self.rho2)
            ids = (loss_sq / prob_grid) * info_gain

            # Mask out greedy and invalid arms
            ids[greedy_idx, :] = np.inf
            for i in range(n):
                if not candidate_mask[i]:
                    ids[i, :] = np.inf

            # Step 5: Pick challenger arm & probability
            ids_min = np.min(ids)
            close_mask = np.abs(ids - ids_min) < self.argmax_tol
            idx_choices = np.argwhere(close_mask)

            if len(idx_choices) == 0:
                challenger_idx, prob_idx = np.unravel_index(np.argmin(ids), ids.shape)
            else:
                selected = self.rng.choice(len(idx_choices))
                challenger_idx, prob_idx = idx_choices[selected]

            p_val = prob_grid[0, prob_idx]
            if self.rng.uniform() < p_val:
                selected_pairs.append([greedy_idx, challenger_idx])
            else:
                selected_pairs.append([greedy_idx, greedy_idx])  # degenerate case

        return selected_pairs
