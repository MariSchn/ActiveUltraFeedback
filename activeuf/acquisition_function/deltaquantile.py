import torch
from activeuf.acquisition_function.base import BaseAcquisitionFunction


class DeltaQuantile(BaseAcquisitionFunction):
    def __init__(
        self, beta: float = 1.0, quantile: float = 0.05, epsilon: float = 0.0, **kwargs
    ):
        """
        Args:
            beta (float): Parameter often used for UCB scaling.
            quantile (float): The center of the rank window (0.0 to 1.0).
            epsilon (float): The half-width of the selection window.
                             The function looks for pairs in the range
                             [quantile - epsilon, quantile + epsilon].
        """
        super().__init__()
        self.beta = beta
        self.quantile = quantile
        self.epsilon = epsilon
        assert 0.0 <= quantile <= 1.0, "Quantile must be between 0.0 and 1.0"
        assert 0.0 <= epsilon <= 1.0, "Epsilon must be between 0.0 and 1.0"
        assert (quantile - epsilon) >= 0.0, "Quantile - Epsilon must be >= 0.0"
        assert (quantile + epsilon) <= 1.0, "Quantile + Epsilon must be <= 1.0"

    def __call__(
        self,
        rewards: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> list[list[int, int]]:
        """
        1. Calculates gap matrix (Upper - Lower).
        2. Sorts gaps.
        3. Identifies a window of candidates around the 'quantile'.
        4. Selects the pair in that window where the first arm has the highest UCB.
        """
        n_prompts, n_completions = upper_bounds.shape
        device = upper_bounds.device

        # 1. Calculate the Gap Matrix
        # Shape: (n_prompts, n_completions, n_completions)
        # Entry [p, i, j] = Upper[p, i] - Lower[p, j]
        confidence_gaps = upper_bounds.unsqueeze(2) - lower_bounds.unsqueeze(1)

        # Mask out diagonal elements (-inf sends them to the bottom of the sort)
        diag_mask = torch.eye(n_completions, device=device, dtype=torch.bool).unsqueeze(
            0
        )
        confidence_gaps.masked_fill_(diag_mask, -torch.inf)

        # Flatten to (n_prompts, n_completions * n_completions)
        gaps_flattened = confidence_gaps.view(n_prompts, -1)

        # --- 2. Define the Index Window ---
        n_valid_pairs = n_completions * (n_completions - 1)

        # Calculate Raw Start Rank
        start_pct = max(0.0, self.quantile - self.epsilon)
        start_rank = int(n_valid_pairs * start_pct)
        # Clamp start to be a valid index (0 to N-1)
        start_rank = min(start_rank, n_valid_pairs - 1)

        # Calculate Raw End Rank
        end_pct = min(1.0, self.quantile + self.epsilon)
        end_rank = int(n_valid_pairs * end_pct)

        if end_rank <= start_rank:
            end_rank = start_rank + 1

        # Final safety: Ensure we don't request more than available pairs
        end_rank = min(end_rank, n_valid_pairs)

        # --- 3. Get Candidate Pairs ---
        # Get top elements up to the end of our window
        _, top_indices = torch.topk(gaps_flattened, k=end_rank, dim=1, sorted=True)

        # Slice to isolate just the window we care about [start : end]
        # If epsilon=0, this slice will have size 1.
        candidate_flat_indices = top_indices[:, start_rank:]

        # --- 4. Select Best First Arm (Highest UCB) ---
        # Convert flat indices to the index of the first arm (i)
        candidate_first_arms = candidate_flat_indices // n_completions

        # Gather UCBs
        candidate_ucbs = torch.gather(upper_bounds, 1, candidate_first_arms)

        # Find index of max UCB within the window
        best_in_window_idx = candidate_ucbs.argmax(dim=1)

        # Retrieve the original flat index
        selected_flat_indices = torch.gather(
            candidate_flat_indices, 1, best_in_window_idx.unsqueeze(1)
        ).squeeze(1)

        # --- 5. Return ---
        first_idxs = selected_flat_indices // n_completions
        second_idxs = selected_flat_indices % n_completions

        return list(zip(first_idxs.tolist(), second_idxs.tolist()))
