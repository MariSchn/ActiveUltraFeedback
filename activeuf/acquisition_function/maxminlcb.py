import numpy as np
import torch
import jax
import jax.numpy as jnp

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class MaxMinLCB(BaseAcquisitionFunction):
    """
    Placeholder for MaxMinLCB acquisition function.
    This class is not implemented yet.
    """

    def __init__(self, beta: float = 1.0, argmax_tol: float = 1e-4, decision_buffer: float = 0.0, use_candidate_set: bool = True, seed: int = 42):
        super().__init__()
        self.beta = beta
        self.argmax_tol = argmax_tol
        self.decision_buffer = decision_buffer
        self.use_candidate_set = use_candidate_set

        if seed is None:
            self.key = jax.random.PRNGKey(0)
        else:
            self.key = jax.random.PRNGKey(seed)

    def __call__(
        self, rewards: torch.Tensor,
        std_deviation: torch.Tensor,
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
            # Shape: (n_completions_per_prompt,)
            rewards_i = rewards[i].cpu().numpy()
            # Shape: (n_completions_per_prompt,)
            std_i = std_deviation[i].cpu().numpy()

            # Create pairwise difference matrix: rewards[i] - rewards[j]
            # Shape: (n_completions, n_completions)
            reward_diff_matrix = rewards_i[:, None] - rewards_i[None, :]

            # Create pairwise uncertainty matrix: std[i] + std[j]
            # Shape: (n_completions, n_completions)
            uncertainty_matrix = std_i[:, None] + std_i[None, :]

            arm_i, arm_j = self._max_min_lcb(
                jnp.array(reward_diff_matrix),
                jnp.array(uncertainty_matrix),
            )
            selected_ids_batch.append((int(arm_i), int(arm_j)))

        return selected_ids_batch

    def _max_min_lcb(
        self, posterior_mean: jnp.ndarray,
            posterior_var: jnp.ndarray
    ) -> tuple[int, int]:
        """
        Computes the max-min LCB acquisition function.

        Args:
            posterior_mean: tensor of shape (n_completions_per_prompt, n_completions_per_prompt)
                containing the posterior means for each completion pair
            posterior_var: tensor of shape (n_completions_per_prompt, n_completions_per_prompt)
                containing the posterior variances for each completion pair

        Returns:
            tuple: Indices of the arms to select.
        """
        lcb = posterior_mean - self.beta * \
            posterior_var  # Shape: (n_arms, n_arms)
        n = lcb.shape[0]

        # Set values to nan for arms that are clearly suboptimal
        if self.use_candidate_set:
            ucb = posterior_mean + self.beta * \
                posterior_var  # Shape: (n_arms, n_arms)
            candidate_arms_mask = jnp.all(
                jnp.logical_or(ucb > -self.decision_buffer,
                               jnp.diag(jnp.full(n, True))),
                axis=1,
            )  # Shape: (n_arms,)
            # Make sure you do not consider the same arms at once, Shape: (n_arms, )
            lcb = jnp.where(
                candidate_arms_mask[:, None] * candidate_arms_mask[None, :],
                lcb,
                jnp.nan,
            )
            lcb = jnp.where(jnp.eye(lcb.shape[0]), jnp.nan, lcb)
        else:
            candidate_arms_mask = jnp.ones(n, dtype=bool)
            lcb = jnp.where(jnp.eye(n), 0, lcb)

        min_j = jnp.nanmin(lcb, axis=1)  # Shape: (n_arms, )
        # argmin_j = jnp.nanargmin(lcb, axis=1)  # Shape: (n_arms, )
        argmin_j_set = jnp.where(
            jnp.abs(lcb - min_j[:, None]) < self.argmax_tol,
            jax.random.choice(self.key, n**2, shape=(n, n), replace=False),
            -jnp.inf,
        )
        argmin_j = jnp.argmax(argmin_j_set, axis=1)
        maxmin_lcb = jnp.nanmax(min_j)  # Shape: ()

        def choose_next_arms():
            argmax_set = jnp.where(
                jnp.abs(min_j - maxmin_lcb) < self.argmax_tol,
                jax.random.choice(self.key, n, shape=(n,), replace=False),
                jnp.nan,
            )
            next_arm_i = jnp.nanargmax(argmax_set)
            next_arm_j = argmin_j[next_arm_i]
            return next_arm_i, next_arm_j

        return jax.lax.cond(
            jnp.sum(candidate_arms_mask) == 1,
            lambda: (
                jnp.nanargmax(candidate_arms_mask),
                jnp.nanargmax(candidate_arms_mask)
            ),
            choose_next_arms,
        )
