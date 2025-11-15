import torch
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
                The order for these is arbitrary and needs to be determined
                using an oracle.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
