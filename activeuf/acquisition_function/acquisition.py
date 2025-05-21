import random
import torch
import json
import argparse
import numpy as np
from abc import ABC, abstractmethod

class BaseAcquisitionFunction(ABC):
    """
    Abstract base class for acquisition functions.
    """
    @abstractmethod
    def __call__(self, *args, **kwargs) -> list[list[int, int]]:
        """
        Given information on the completions for a batch of prompts, selects 
        the indices for the two completions per prompt that should be annotated
        by the oracle.

        Args:
            Blank, because it can vary across the child classes.

        Returns:
            list[list[int, int]]: The selected indices per prompt. 
                The first index is the one that should be chosen, the second 
                one is the one that should be rejected.
        """
        pass

class RandomAcquisitionFunction(BaseAcquisitionFunction):
    """
    Randomly selects and returns two indices per prompt
    """

    def __init__(self):
        super().__init__()

    def __call__(
            self, 
            n_prompts: int, 
            n_completions_per_prompt: int,
        ) -> list[list[int, int]]:
        """
        Args:
            n_prompts: number of prompts
            n_completions_per_prompt: number of completions per prompt

        Returns:
            list[list[int, int]]: The selected indices per prompt.
                The first index is the one that should be chosen, the second 
                one is the one that should be rejected.
        """
        return np.random.randint(
            low=0,
            high=n_completions_per_prompt,
            size=(n_prompts, 2),
        ).tolist()
    
    
class DoubleThompsonSampling(BaseAcquisitionFunction):
    def __init__(self, max_iterations: int = 10, beta: int = 1):
        super().__init__()
        self.max_iterations = max_iterations
        self.beta = beta
        
    def __call__(
        self, 
        rewards: torch.Tensor, 
        lower_bounds: torch.Tensor, 
        upper_bounds: torch.Tensor,
    ) -> list[list[int, int]]:
        """
        Args:
            rewards: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the reward scores for each completion
            lower_bounds: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the lower bounds for each completion
            upper_bounds: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the upper bounds for each completion
        Returns:
            list[list[int, int]]: The selected indices per prompt.
                The first index is the one that should be chosen, the second 
                one is the one that should be rejected.
        """

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

    def dts_optimize(self, reward_list, lower_bound_list, upper_bound_list):
        r_epistemic_index = []
        for j in range(len(reward_list)):
            z = np.random.uniform(-1, 1)
            difference_half = (upper_bound_list[j] - lower_bound_list[j]) / 2
            reward_middle = lower_bound_list[j] + difference_half

            r_x_y_epistemic_index = reward_middle + self.beta * z * difference_half
            r_epistemic_index.append(r_x_y_epistemic_index)
        return np.argmax([idx.cpu() for idx in r_epistemic_index])
    