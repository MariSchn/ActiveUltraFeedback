from abc import ABC, abstractmethod
import torch
import random

import regex as re


def init_oracle(oracle_name: str):
    """
    Parses the oracle class name and returns the corresponding oracle class.

    Args:
        oracle_name (str): The name of the oracle class to parse.

    Returns:
        BaseOracle: The corresponding oracle class.
    """
    if oracle_name.lower() == "random":
        return RandomOracle()
    elif oracle_name.lower() == "ultrafeedback":
        return UltraFeedbackOracle()
    else:
        raise ValueError(f"Unknown oracle class: {oracle_name}")


class BaseOracle(ABC):
    """
    This is the base class for all oracles. It defines the interface that all oracles must implement.
    The task of oracles is: Given 2 completions for the same prompt, select which one is the chosen and which one is the rejected one.
    """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(
        self, prompts_with_completions: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        This function should be overridden by subclasses to implement the specific oracle logic.
        The oracle takes prompts with two completions and selects which completion is the chosen and which is the rejected one.

        Args:
            prompts_with_completions (list[dict[str, str]]): A list of dictionaries, each containing a prompt and 2 completions.
                Each dictionary should have the following keys:
                - "prompt": The prompt text.
                - "prompt_id": The prompt id.
                - "response_text_1": The first completion text.
                - "model_1": The model of the first completion.
                - "response_text_2": The second completion text.
                - "model_2": The model of the second completion.
        Returns:
            list[dict[str, str]]: A list of dictionaries, each containing a sample.
                Each dictionary should have the following keys
                - "prompt": The prompt text.
                - "prompt_id": The prompt id.
                - "chosen": The chosen completion text.
                - "chosen_model": The model of the chosen completion.
                - "rejected": The rejected completion text.
                - "rejected_model": The model of the rejected completion.
        """
        pass


class RandomOracle(BaseOracle):
    """
    This oracle randomly selects among the two passed completions which one is the chosen and which one is the rejected one.
    It is mainly used for debugging purposes and as a baseline.
    """

    def __init__(self):
        super().__init__()

    def __call__(
        self, prompts_with_completions: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Rnadomly selects among the two passed completions which one is the chosen and which one is the rejected one.

        Args:
            prompts_with_completions (list[dict[str, str]]): A list of dictionaries, each containing a prompt and 2 completions.
                Each dictionary should have the following keys:
                - "prompt": The prompt text.
                - "prompt_id": The prompt id.
                - "response_text_1": The first completion text.
                - "model_1": The model of the first completion.
                - "response_text_2": The second completion text.
                - "model_2": The model of the second completion.
        Returns:
            list[dict[str, str]]: A list of dictionaries, each containing a sample.
                Each dictionary should have the following keys
                - "prompt": The prompt text.
                - "prompt_id": The prompt id.
                - "chosen": The chosen completion text.
                - "chosen_model": The model of the chosen completion.
                - "rejected": The rejected completion text.
                - "rejected_model": The model of the rejected completion.
        """
        out = []

        for x in prompts_with_completions:
            chosen_int, rejected_int = 1, 2
            if random.random() > 0.5:
                chosen_int, rejected_int = rejected_int, chosen_int
            out.append(
                {
                    "prompt": x["prompt"],
                    "prompt_id": x["prompt_id"],
                    "chosen": x[f"response_text_{chosen_int}"],
                    "chosen_model": x[f"model_{chosen_int}"],
                    "chosen_score": x[f"overall_score_{chosen_int}"],
                    "rejected": x[f"response_text_{rejected_int}"],
                    "rejected_model": x[f"model_{rejected_int}"],
                    "rejected_score": x[f"overall_score_{rejected_int}"],
                }
            )
        return out


class UltraFeedbackOracle(BaseOracle):
    """
    This oracle implements the annotation approach proposed in the paper https://arxiv.org/abs/2310.01377.
    It uses a LLM as a judge to annotate the completions for multiple aspects.
    The completion with the highest overall score is selected as the chosen one, and the other one is selected as the rejected one.
    """

    def __init__(self):
        super().__init__()

    def parse_score_str(self, score_str: str) -> float:
        if isinstance(score_str, str):
            return float(score_str)
        elif isinstance(score_str, int):
            return float(score_str)
        elif isinstance(score_str, float):
            return float(score_str)
        elif isinstance(score_str, torch.Tensor):
            return float(score_str.item())
        else:
            raise ValueError(
                f"Unknown type {type(score_str)} for score_str: {score_str}. Expected str, int, float or torch.Tensor."
            )

    def parse_score_str_old(self, score_str: str) -> int:
        try:
            match = re.search(r"(\d+)", score_str)
            score = int(match.group())
            score = max(0, min(score, 10))
            return score
        except:
            print(f"Could not parse score from {score_str}")
            return 0

    def __call__(
        self, prompts_with_completions: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Selects among the two passed completions which one is the chosen and which one is the rejected one.

        Args:
            prompts_with_completions (list[dict[str, str]]): A list of dictionaries, each containing a prompt and 2 completions.
                Each dictionary should have the following keys:
                - "prompt": The prompt text.
                - "prompt_id": The prompt id.
                - "response_text_1": The first completion text.
                - "score_1": The score of the first completion.
                - "model_1": The model of the first completion.
                - "response_text_2": The second completion text.
                - "model_2": The model of the second completion.
                - "score_2": The score of the second completion.
        Returns:
            list[dict[str, str]]: A list of dictionaries, each containing a sample.
                Each dictionary should have the following keys
                - "prompt": The prompt text.
                - "prompt_id": The prompt id.
                - "chosen": The chosen completion text.
                - "chosen_model": The model of the chosen completion.
                - "chosen_score": The overall score of the chosen completion.
                - "input_ids_chosen": The input ids of the chosen completion.
                - "attention_mask_chosen": The attention mask of the chosen completion.
                - "rejected": The rejected completion text.
                - "rejected_model": The model of the rejected completion.
                - "rejected_score": The overall score of the rejected completion.
                - "input_ids_rejected": The input ids of the rejected completion.
                - "attention_mask_rejected": The attention mask of the rejected completion.
        """
        out = []
        for x in prompts_with_completions:
            chosen_int, rejected_int = 1, 2
            if self.parse_score_str(x["score_1"]) < self.parse_score_str(x["score_2"]):
                chosen_int, rejected_int = 2, 1

            
            out.append({
                "prompt": x["prompt"],
                "prompt_id": x["prompt_id"],

                "chosen": x[f"response_text_{chosen_int}"],
                "chosen_model": x[f"model_{chosen_int}"],
                "chosen_score": x[f"score_{chosen_int}"],
                "input_ids_chosen": x.get(f"input_ids_{chosen_int}"),
                "attention_mask_chosen": x.get(f"attention_mask_{chosen_int}"),
                "features_chosen": x.get(f"features_{chosen_int}"),

                "rejected": x[f"response_text_{rejected_int}"],
                "rejected_model": x[f"model_{rejected_int}"],
                "rejected_score": x[f"score_{rejected_int}"],
                "input_ids_rejected": x.get(f"input_ids_{rejected_int}"),
                "attention_mask_rejected": x.get(f"attention_mask_{rejected_int}"),
                "features_rejected": x.get(f"features_{rejected_int}"),
            })
        return out
