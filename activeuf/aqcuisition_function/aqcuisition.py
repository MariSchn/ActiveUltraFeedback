import random
import torch
import json
import argparse
import numpy as np

class AcquisitionFunction():
    """
    Abstract base class for acquisition functions.
    """
    def select(self, rewards: torch.Tensor, uncertainty: torch.Tensor) -> list:
        """
        Selects indices of prompts/completions for annotation.

        Args:
            rewards (torch.Tensor): Rewards tensor with shape (n_prompts, n_completions)
            uncertainty (torch.Tensor): Uncertainty tensor with shape (n_prompts, n_completions)

        Returns:
            List[int]: List of indices of selected prompts/completions
        """
        pass

class RandomAcquisitionFunction(AcquisitionFunction):
    """
    Selects the highest and lowest reward completions per prompt.
    Returns a list of (prompt_idx, chosen_idx, rejected_idx) tuples.
    """

    def __init__(self):
        pass

    def select(self, rewards: list, uncertainty: list) -> list:
        """
        rewards: List of lists (prompt x completions)
        uncertainty: List of lists (same shape, not used here)

        Returns:
            List of tuples (prompt_idx, chosen_idx, rejected_idx)
        """
        selected_triplets = []

        for i, reward_list in enumerate(rewards):
            if len(reward_list) < 2:
                continue  # Need at least 2 completions to compare

            # Find indices of max and min reward
            max_idx = reward_list.index(max(reward_list))
            min_idx = reward_list.index(min(reward_list))

            selected_triplets.append((i, max_idx, min_idx))

        return selected_triplets