import torch
import numpy as np
from abc import ABC, abstractmethod

from activeuf.acquisition_function.random import RandomAcquisitionFunction
from activeuf.acquisition_function.dts import DoubleThompsonSampling
from activeuf.acquisition_function.infomax import InfoMax


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


def init_acquisition_function(
    acquisition_type: str,
    **kwargs,
) -> BaseAcquisitionFunction:
    """
    Initializes the acquisition function based on the specified type.

    Args:
        acquisition_type (str): The type of acquisition function to initialize.

    Returns:
        BaseAcquisitionFunction: An instance of the specified acquisition function.
    """
    if acquisition_type == "random":
        return RandomAcquisitionFunction(**kwargs)
    elif acquisition_type == "double_thompson_sampling":
        return DoubleThompsonSampling(**kwargs)
    elif acquisition_type == "infomax":
        return InfoMax(**kwargs)
    else:
        raise ValueError(
            f"Unknown acquisition function type: {acquisition_type}")
