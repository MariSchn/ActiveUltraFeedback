import datasets
import json
import pandas as pd
import torch
from typing import Dict, Any

from fastchat import conv_template
from principles import principles

import os
import argparse

import random
import numpy as np
import torch

from vllm import LLM, SamplingParams

"""
This script is used to generate completions for one dataset using one model with vLLM.
To generate the completions for all models, this script needs to be run multiple times, once for each model.
Before this script is run, the dataset need to be created using the `create_raw_dataset.py` script.

To run it, you need to pass the model (`model_name`) and the dataset (`dataset`) as arguments. 
The `model_name` is the name of the model to use for completions (e.g. llama-2-13b-chat) and the `dataset` is the name of the dataset to download and process (e.g. truthful_qa).
To map the inputs to the actual model path or Huggingface model the `model_path` dictionary is used and needs to be updated if you want to use a new model.
"""

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# TODO: Add (Huggingface) paths for the remaining models in the model_pool
model_map = {
    "ultralm-13b": "openbmb/UltraLM-13b-v2.0",
    "gpt-2": "openai-community/gpt2"  # ! Only used for testing, as it is a relatively small model that can be easily loaded
}

def load_generator(model: str) -> LLM:
    """
    Generates a vLLM generator object for the given model.

    Args:
        model: The Huggingface model path to load the generator for, e.g. "openbmb/UltraLM-13b-v2.0"
    Returns:
        The loaded generator object
    """
    dtype = "auto" if model_name not in ["starchat", "mpt-30b-chat", "falcon-40b-instruct"] else "bfloat16"
    gpu_memory_utilization = 0.95
    model = LLM(model, gpu_memory_utilization=gpu_memory_utilization, swap_space=1, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, dtype=dtype)

    return model

@torch.no_grad()
def instruction_completion(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Completes one individual instruction example.
    This is supposed to be used with the `map` function from the Huggingface datasets library to generate completions for a dataset.

    Args:
        example (dict): A dictionary containing the instruction and models to be used for completion. 
                        It should at least have the following keys:
                        - "instruction": The instruction to be completed.
                        - "models": A list of model types that can be used for completion.
                        - "completions": A list to store the generated completions.
    Returns:
        dict: The updated example dictionary with the generated completion appended to the "completions" list.
    """
    # Skip if this example should not be completed by this model
    if model_name not in example["models"]:
        return example
    
    # Set principle
    if dataset_name in ["sharegpt"]:
        principle = random.choice(["helpfulness", "helpfulness", "helpfulness", "truthfulness", "honesty"])
    elif dataset_name in ["ultrachat"]:
        principle = random.choice(["helpfulness", "helpfulness", "helpfulness", "truthfulness", "honesty"])
    elif dataset_name in ["flan"]:
        principle = random.choice(["helpfulness", "helpfulness", "helpfulness", "helpfulness", "verbalized_calibration"])
    elif dataset_name in ["evol_instruct"]:
        principle = "helpfulness"
    elif dataset_name in ["truthful_qa", "false_qa"]:
        principle = random.choice(["honesty", "truthfulness"])
    else:
        print(f"No principle defined for subset {dataset_name}. Falling back to helpfulness.")
        principle = "helpfulness"

    if principle == "honesty":
        principle = "honesty" if np.random.rand() < 0.9 else "verbalized_calibration"

    principle_prompt = random.choice(principles[principle])

    # Set generation format
    # TODO: Adapt for more models (currently just gpt-2 is tested)
    if "ultralm" in model_name:
        system_prompt = "User: A one-turn chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, very detailed, and polite answers to the user's questions.</s>"
        system_prompt += "User: " + principle_prompt + "</s>"
        conv = [system_prompt]
        conv.append("User: " + example["instruction"] + "</s>")
        conv.append("Assistant: ")
        prompt = "\n".join(conv)
    elif "starchat" in model_name:
        system_prompt = "<|system|>" + principle_prompt + "<|end|>"
        conv = [system_prompt]
        conv.append("<|user|>" + example["instruction"] + "<|end|>")
        conv.append("<|assistant|>")
        prompt = "\n".join(conv)
    elif model_name == "wizardlm-7b":
        prompt = "{}\n\n### Response:".format(example["instruction"])
    elif model_name.split("-")[0] in ["llama", "alpaca", "vicuna", "mpt", "falcon", "wizardlm"]: # note that the wizardlm should be 13b or 30b
        conv = conv_template[model_name.split("-")[0]].copy()
        conv.system += " " + principle_prompt
        conv.append_message(conv.roles[0], example["instruction"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif model_name == "gpt-2":  # Only used for testing, as gpt-2 is a relatively small model that can be easily loaded
        prompt = example["instruction"]
    else:
        raise NotImplementedError
    
    # Generate completion
    with torch.inference_mode():
        # TODO: Adapt for more models (currently just gpt-2 is tested)
        if model_name.split("-")[0] in ["llama", "alpaca", "vicuna", "mpt", "falcon", "wizardlm"]:
            conv = conv_template[model_name.split("-")[0]].copy()
            if conv.stop_str is not None:
                stop = [conv.stop_str]
            elif conv.stop_token_ids is not None:
                stop = [generator.llm_engine.tokenizer.decode(stop_token_id) for stop_token_id in conv.stop_token_ids]
            else: # ultralm
                stop = ["</s>"]
        else: # ultralm
            stop = ["</s>"]

        sampling_params = SamplingParams(temperature=1, top_p=1, max_tokens=max_tokens, stop=stop)

        responses = generator.generate(prompt, sampling_params)
        responses = [response.outputs[0].text.strip().rstrip("</s>").strip() for response in responses]
    
    example["completions"].append({
        "model": model_name,
        "principle": principle,
        "custom_system_prompt": principle_prompt,
        "response": responses[0]
    })

    return example


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use for completions (e.g. llama-2-13b-chat)")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to load (e.g. truthful_qa)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="The maximum number of tokens to generate for each completion")
    parser.add_argument("--seed", type=int, default=123, help="Seed for random sampling")
    args = parser.parse_args()

    # Set random seed
    if isinstance(args.seed, int):
        random.seed(args.seed)

    model_name = args.model_name
    dataset_name = args.dataset_name
    max_tokens = args.max_tokens

    # Load generator
    generator = load_generator(model_map[model_name])

    # Load dataset
    print(f"Loading dataset {dataset_name}")
    load_path = f"./completion_data/{dataset_name}.json"
    dataset = json.load(open(load_path))
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(dataset))

    # Generate completions
    dataset = dataset.map(instruction_completion, desc=f"{model_name} on {dataset_name} generating completions")

    # Save dataset with completions
    result_path = load_path
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump([{k: v for k, v in data.items()} for data in dataset], f, indent=4)
