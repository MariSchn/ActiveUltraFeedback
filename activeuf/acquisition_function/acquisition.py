import random
import torch
import json
import argparse
import numpy as np
from abc import ABC, abstractmethod

class AcquisitionFunction(ABC):
    """
    Abstract base class for acquisition functions.
    """
    @abstractmethod
    def __call__(self, rewards: torch.Tensor, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor) -> list:
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
    Selects the two random completions per prompt.
    Returns a list of (response_1_idx, response_2_idx) tuples chosen uniformly random. 
    """

    def __init__(self):
        super().__init__()

    def __call__(self, rewards: torch.Tensor, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor) -> list:
        """
        rewards: List of lists (prompt x completions)
        uncertainty: List of lists (same shape, not used here)

        Returns:
            List of tuples (response_1_idx, response_2_idx)
        """
        selected_indices = []

        for reward_list in rewards:
            if len(reward_list) < 2:
                raise Exception("Need at least 2 completions for the aquisition function")

            random_indices = random.sample(range(len(reward_list)), 2)

            selected_indices.append((random_indices[0], random_indices[1]))

        return selected_indices
    
    
class DoubleThompsonSampling(AcquisitionFunction):
    def __init__(self, beta=1, max_iterations=10):
        super().__init__()
        self.max_iterations = max_iterations
        self.beta = beta
        
    def __call__(self, rewards: torch.Tensor, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor) -> list:
        selected_ids_batch = []
        for i in range(len(rewards)):
            #step 1 - selecting first response
            response_1 = self.dts_optimize(rewards[i], lower_bounds[i], upper_bounds[i])
            
            #step 2 - selecting second response
            response_2 = response_1
            iterations = 0
            while response_1 == response_2:
                if iterations == self.max_iterations:
                    response_2 = np.random.randint(0, len(rewards[i]))
                else:
                    response_2 = self.dts_optimize(rewards[i], lower_bounds[i], upper_bounds[i])
                    iterations += 1
            
            selected_ids_batch.append((response_1, response_2))

        return selected_ids_batch
        
    """
        For beta=1, we uniformly sample the reward from the interval [lower_bound; upper_bound].
    """
    def dts_optimize(self, reward_list, lower_bound_list, upper_bound_list):
        r_epistemic_index = []
        for j in range(len(reward_list)):
            z = np.random.uniform(-1, 1)
            difference_half = (upper_bound_list[j] - lower_bound_list[j]) / 2
            reward_middle = lower_bound_list[j] + difference_half

            r_x_y_epistemic_index = reward_middle + self.beta * z * difference_half
            r_epistemic_index.append(r_x_y_epistemic_index)
        return np.argmax(r_epistemic_index)
    