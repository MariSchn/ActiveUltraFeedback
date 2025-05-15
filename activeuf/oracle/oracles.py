import random
from typing import List, Dict

import numpy as np
import regex as re

from datasets import Dataset, load_from_disk
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
                - "prompt_id": The prompt id.
                - "completion_1": The first completion text.
                - "model_1": The model of the first completion.
                - "completion_2": The second completion text.
                - "model_2": The model of the second completion.
        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing a sample.
                Each dictionary should have the following keys
                - "prompt": The prompt text.
                - "prompt_id": The prompt id.
                - "chosen": The chosen completion text.
                - "chosen_model": The model of the chosen completion.
                - "rejected": The rejected completion text.
                - "rejected_model": The model of the rejected completion.
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
                - "prompt_id": The prompt id.
                - "completion_1": The first completion text.
                - "model_1": The model of the first completion.
                - "completion_2": The second completion text.
                - "model_2": The model of the second completion.
        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing a sample.
                Each dictionary should have the following keys
                - "prompt": The prompt text.
                - "prompt_id": The prompt id.
                - "chosen": The chosen completion text.
                - "chosen_model": The model of the chosen completion.
                - "rejected": The rejected completion text.
                - "rejected_model": The model of the rejected completion.
        """
        annotated_samples = []

        for sample in prompts_with_completions:
            if random.random() < 0.5:
                annotated_samples.append({
                    "prompt": sample["prompt"],
                    "prompt_id": sample["prompt_id"],
                    "chosen": sample["completion_1"],
                    "chosen_model": sample["model_1"],
                    "rejected": sample["completion_2"],
                    "rejected_model": sample["model_2"],
                })
            else:
                annotated_samples.append({
                    "prompt": sample["prompt"],
                    "prompt_id": sample["prompt_id"],
                    "chosen": sample["completion_2"],
                    "chosen_model": sample["model_2"],
                    "rejected": sample["completion_1"],
                    "rejected_model": sample["model_1"],
                })
        
        return annotated_samples
    
class UltraFeedbackOracle(BaseOracle):
    """
    This oracle implements the annotation approach proposed in the paper https://arxiv.org/abs/2310.01377.
    It uses a LLM as a judge to annotate the completions for multiple aspects.
    The completion with the highest overall score is selected as the chosen one, and the other one is selected as the rejected one.
    """
    def __init__(self):
        super().__init__()

    def parse_overall_score_str(self, overall_score_str: str) -> int:
        try:
            match = re.search(r"(\d+)", overall_score_str)
            score = int(match.group())
            score = max(0, min(score, 10))
            return score
        except:
            print(f"Could not parse overall score from {overall_score_str}")
            return 0

    def __call__(self, prompts_with_completions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Selects among the two passed completions which one is the chosen and which one is the rejected one.
        
        Args:
            prompts_with_completions (List[Dict[str, str]]): A list of dictionaries, each containing a prompt and 2 completions.
                Each dictionary should have the following keys:
                - "prompt": The prompt text.
                - "prompt_id": The prompt id.
                - "completion_1": The first completion text.
                - "model_1": The model of the first completion.
                - "completion_2": The second completion text.
                - "model_2": The model of the second completion.
        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing a sample.
                Each dictionary should have the following keys
                - "prompt": The prompt text.
                - "prompt_id": The prompt id.
                - "chosen": The chosen completion text.
                - "chosen_model": The model of the chosen completion.
                - "rejected": The rejected completion text.
                - "rejected_model": The model of the rejected completion.
        """
        # Use pre-computed annotations if available
        binarized_batch = []

        # TODO Maybe process a batch at once instead of one by one
        for sample in prompts_with_completions:
            overall_score_1 = self.parse_overall_score_str(sample["overall_score_1"])
            overall_score_2 = self.parse_overall_score_str(sample["overall_score_2"])

            binarized_sample = {
                "prompt": sample["prompt"],
                "chosen": sample["completion_1"] if overall_score_1 > overall_score_2 else sample["completion_2"],
                "rejected": sample["completion_2"] if overall_score_1 > overall_score_2 else sample["completion_1"],
            }
            binarized_batch.append(binarized_sample)

        return binarized_batch