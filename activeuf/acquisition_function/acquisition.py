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
    Returns a list of (chosen_idx, rejected_idx) tuples.
    """

    def __init__(self):
        pass

    def select(self, rewards: list, uncertainty: list) -> list:
        """
        rewards: List of lists (prompt x completions)
        uncertainty: List of lists (same shape, not used here)

        Returns:
            List of tuples (chosen_idx, rejected_idx)
        """
        selected_indices = []

        for reward_list in rewards:
            if len(reward_list) < 2:
                raise Exception("Need at least 2 completions for the aquisition function")

            # Find indices of max and min reward
            max_idx = reward_list.index(max(reward_list))
            min_idx = reward_list.index(min(reward_list))

            selected_indices.append((max_idx, min_idx))

        return selected_indices
    
    
class DoubleThompsonSampling(AcquisitionFunction):
    def __init__(self, max_iterations=10):
        super().__init__()
        self.max_iterations = max_iterations
        
    def select(self, rewards: list, uncertainties: list) -> list:
        selected_ids_batch = []
        for i in range(len(rewards)):
            #step 1 - selecting first response
            response_1 = self.dto_optimize(rewards[i], uncertainties[i])
            
            #step 2 - selecting second response
            response_2 = response_1
            iterations = 0
            while response_1 == response_2:
                if iterations == self.max_iterations:
                    response_2 = np.random.randint(0, len(rewards[i]))
                else:
                    response_2 = self.dto_optimize(rewards[i], uncertainties[i])
                    iterations += 1
            
            selected_ids_batch.append((response_1, response_2))

        return selected_ids_batch
        
    def dto_optimize(self, reward_list, uncertainty_list):
        r_epistemic_index = []
        for j in range(len(reward_list)):
            r_x_y_epistemic_index = np.random.normal(reward_list[j], np.sqrt(uncertainty_list[j]))
            r_epistemic_index.append(r_x_y_epistemic_index)
        return np.argmax(r_epistemic_index)
    