import torch
from activeuf.acquisition_function.base import BaseAcquisitionFunction


class DeltaQuantile(BaseAcquisitionFunction):
    def __init__(self, beta: float = 1.0, quantile: float = 0.1, **kwargs):
        """
        Args:
            beta (float): Parameter often used for UCB scaling (kept for compatibility).
            quantile (float): The specific rank to select from the sorted pairs.
                              0.0 = Top 1 (max gap).
                              1.0 = Bottom (min gap among valid pairs).
                              0.1 = The element at the top 10% mark.
        """
        super().__init__()
        self.beta = beta
        self.quantile = quantile

    def __call__(
        self,
        rewards: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> list[list[int, int]]:
        """
        Selects a pair of indices (i, j) for each prompt such that the gap
        (upper_bounds[i] - lower_bounds[j]) represents the 'quantile' percentile
        of all valid pairs.
        """
        n_prompts, n_completions = upper_bounds.shape
        device = upper_bounds.device

        # 1. Calculate the Gap Matrix
        # Shape: (n_prompts, n_completions, n_completions)
        # Entry [p, i, j] = Upper[p, i] - Lower[p, j]
        confidence_gaps = upper_bounds.unsqueeze(2) - lower_bounds.unsqueeze(1)

        # 2. Mask out diagonal elements
        # We compare i vs j. We should not compare i vs i.
        # We fill diagonals with -inf so they appear at the very end of a descending sort.
        diag_mask = torch.eye(n_completions, device=device, dtype=torch.bool).unsqueeze(
            0
        )
        confidence_gaps.masked_fill_(diag_mask, -torch.inf)

        # 3. Flatten to (n_prompts, n_completions * n_completions)
        gaps_flattened = confidence_gaps.view(n_prompts, -1)

        # 4. Calculate the target index based on the quantile parameter
        # Total valid pairs per prompt = N * (N - 1)
        n_valid_pairs = n_completions * (n_completions - 1)

        # If quantile is 0.15, we want the index at 15% of the way down the sorted list.
        # We clamp to ensure we don't go out of bounds (0 to n_valid_pairs - 1).
        target_rank = int(n_valid_pairs * self.quantile)
        target_rank = max(0, min(target_rank, n_valid_pairs - 1))

        # 5. Retrieve the element at the target rank
        # We use topk with k = target_rank + 1. The last column of the result
        # is the element at 'target_rank'.
        # Note: -inf values (diagonals) are at the end, so as long as
        # target_rank < n_valid_pairs, we will never select a diagonal.
        _, top_indices = torch.topk(
            gaps_flattened, k=target_rank + 1, dim=1, sorted=True
        )

        # The last column corresponds to the specific quantile we want
        selected_flat_indices = top_indices[:, -1]

        # 6. Convert flat indices back to (i, j) pairs
        first_idxs = selected_flat_indices // n_completions
        second_idxs = selected_flat_indices % n_completions

        return list(zip(first_idxs.tolist(), second_idxs.tolist()))
