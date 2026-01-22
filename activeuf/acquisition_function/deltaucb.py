import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class DeltaUCB(BaseAcquisitionFunction):
    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__()
        self.beta = beta

    def __call__(
        self,
        rewards: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> list[list[int, int]]:
        n_prompts, n_completions = upper_bounds.shape

        # Shape: (n_prompts, n_completions, n_completions)
        upper_confidence_bounds = upper_bounds.unsqueeze(2) - lower_bounds.unsqueeze(1)

        # Mask out diagonal to avoid choosing the same completion twice
        diag_mask = torch.eye(
            n_completions, device=upper_confidence_bounds.device, dtype=torch.bool
        ).unsqueeze(0)
        upper_confidence_bounds.masked_fill_(diag_mask, -torch.inf)

        # Flatten the n_completions x n_completions matrices, find argmax and convert back to (i, j) pairs)
        upper_confidence_bounds_flattened = upper_confidence_bounds.view(n_prompts, -1)
        max_confidence_gap_flat_idx = upper_confidence_bounds_flattened.argmax(dim=1)

        first_idxs = max_confidence_gap_flat_idx // n_completions
        second_idxs = max_confidence_gap_flat_idx % n_completions

        return list(zip(first_idxs.tolist(), second_idxs.tolist()))
