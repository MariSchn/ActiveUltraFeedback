import random
from typing import List, Dict

import numpy as np

from datasets import Dataset
from activeuf.annotate_completions import annotate

def parse_oracle_class(oracle_class: str):
    """
    Parses the oracle class name and returns the corresponding oracle class.
    
    Args:
        oracle_class (str): The name of the oracle class to parse.
        
    Returns:
        BaseOracle: The corresponding oracle class.
    """
    if oracle_class.lower() == "random":
        return RandomOracle
    elif oracle_class.lower() == "ultrafeedback":
        return UltraFeedbackOracle
    else:
        raise ValueError(f"Unknown oracle class: {oracle_class}")

class BaseOracle:
    """
    This is the base class for all oracles. It defines the interface that all oracles must implement.
    The task of oracles is: Given 2 completions for the same prompt, select which one is the chosen and which one is the rejected one.
    """
    def __init__(self):
        pass

    def __call__(self, prompts_with_completions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        This function should be overridden by subclasses to implement the specific oracle logic.
        The oracle takes prompts with two completions and selects which completion is the chosen and which is the rejected one.

        Args:
            prompts_with_completions (List[Dict[str, str]]): A list of dictionaries, each containing a prompt and 2 completions.
                Each dictionary should have the following keys:
                - "prompt": The prompt text.
                - "completion_1": The first completion text.
                - "completion_2": The second completion text.
        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing a sample.
                Each dictionary should have the following keys
                - "prompt": The prompt text.
                - "chosen": The chosen completion text.
                - "rejected": The rejected completion text.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

class RandomOracle(BaseOracle):
    """
    This oracle randomly selects among the two passed completions which one is the chosen and which one is the rejected one.
    It is mainly used for debugging purposes and as a baseline.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, prompts_with_completions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Rnadomly selects among the two passed completions which one is the chosen and which one is the rejected one.
        
        Args:
            prompts_with_completions (List[Dict[str, str]]): A list of dictionaries, each containing a prompt and 2 completions.
                Each dictionary should have the following keys:
                - "prompt": The prompt text.
                - "completion_1": The first completion text.
                - "completion_2": The second completion text.
        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing a sample.
                Each dictionary should have the following keys
                - "prompt": The prompt text.
                - "chosen": The chosen completion text.
                - "rejected": The rejected completion text.
        """
        annotated_samples = []

        for sample in prompts_with_completions:
            chosen, rejected = random.sample([sample["completion_1"], sample["completion_2"]], 2)

            annotated_sample = {
                "prompt": sample["prompt"],
                "chosen": chosen,
                "rejected": rejected
            }
            annotated_samples.append(annotated_sample)

        return annotated_samples
    
class UltraFeedbackOracle(BaseOracle):
    """
    This oracle implements the annotation approach proposed in the paper https://arxiv.org/abs/2310.01377.
    It uses a LLM as a judge to annotate the completions for multiple aspects.
    The completion with the highest overall score is selected as the chosen one, and the other one is selected as the rejected one.
    """
    def __init__(self, dataset_cache: Dataset | None = None, output_path: str | None = None):
        """
        Initializes the UltarFeedbackOracle.

        Pass a dataset_cache if you want to cache the annotations the oracle generates.

        Args:
            dataset_cache (Dataset, optional): A Hugging Face Dataset to cache the annotations. Defaults to None.
            output_path (str, optional): The path to save the dataset cache. If using a dataset_cache, this is required. Defaults to None.
        """
        super().__init__()

        if dataset_cache is not None and output_path is None:
            raise ValueError("If using a dataset_cache, you must provide an output_path to save it.")

        self.dataset_cache = dataset_cache
        self.output_path = output_path

    def __call__(self, prompts_with_completions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Selects among the two passed completions which one is the chosen and which one is the rejected one.
        
        Args:
            prompts_with_completions_for_annotation (List[Dict[str, str]]): A list of dictionaries, each containing a prompt and 2 completions.
                Each dictionary should have the following keys:
                - "prompt": The prompt text.
                - "completion_1": The first completion text.
                - "completion_2": The second completion text.
        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing a sample.
                Each dictionary should have the following keys
                - "prompt": The prompt text.
                - "chosen": The chosen completion text.
                - "rejected": The rejected completion text.
        """
        raise NotImplementedError("TODO: Implement the UltraFeedbackOracle")