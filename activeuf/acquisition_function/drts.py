import numpy as np
import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class DoubleReverseThompsonSampling(BaseAcquisitionFunction):
    def __init__(self, max_iterations: int = 30, beta: int = 1, **kwargs):
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
            std_deviation: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the standard deviation of the reward for each completions
        Returns:
            list[list[int, int]]: The selected indices per prompt.
                The order for these is arbitrary and needs to be determined
                using an oracle.
        """
        # even if upper and lower bounds are not symmetric, our current implementation is such, that we can assume they are symmetric.
        std_deviation = (upper_bounds - lower_bounds) / 2

        selected_ids_batch = []
        for i in range(len(rewards)):
            # step 1 - selecting first response
            response_1 = self.dts_optimize(rewards[i], std_deviation[i])

            # step 2 - selecting second response
            response_2 = response_1
            iterations = 0
            while response_1 == response_2:
                if iterations == self.max_iterations:
                    response_2 = np.random.randint(0, len(rewards[i]))
                else:
                    response_2 = self.dts_reverse_optimize(rewards[i], std_deviation[i])
                    iterations += 1

            selected_ids_batch.append((response_1, response_2))

        return selected_ids_batch

    def dts_optimize(self, reward_list, std_deviation_list):
        r_epistemic_index = []
        for j in range(len(reward_list)):
            z = np.random.uniform(-1, 1)

            r_x_y_epistemic_index = (
                reward_list[j] + self.beta * z * std_deviation_list[j]
            )
            r_epistemic_index.append(r_x_y_epistemic_index)
        return np.argmax([idx.cpu() for idx in r_epistemic_index])

    def dts_reverse_optimize(self, reward_list, std_deviation_list):
        # negating all the reward estimates and passing them to dts_optimize
        negated_reward_list = [-r for r in reward_list]
        return self.dts_optimize(negated_reward_list, std_deviation_list)
