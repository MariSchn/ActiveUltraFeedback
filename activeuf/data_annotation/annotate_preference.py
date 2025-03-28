
import requests
import time
import datasets
import json
import pandas as pd
import random
import argparse
from typing import List, Dict

import os
import re
from copy import deepcopy
from tqdm import tqdm

from prompts import prompt_templates, system_prompt

import openai
openai.api_key = "PUT YOUR KEY HERE"

"""
This script annotates the completions in a dataset according to the following aspects:
- Instruction following
- Honesty
- Truthfulness
- Helpfulness

The annotations are stored in the "annotations" field of each completion.
For this to work the dataset already needs to contain completions, which can be generated using the `main_vllm.py` and `create_raw_dataset.py` scripts 
in the `comparison_data_generation` directory.
"""

ASPECTS = ["instruction_following", "honesty", "truthfulness", "helpfulness"]

def process(responses: str, aspect: str) -> List[Dict[str, str]]:
    """
    Processes the response from an evaluation/annotation model and extracts the ratings and rationales for each completion from the response.
    As it is not always guaranteed that the response follows the expected format, the function will raise an error if the response does not follow the expected format.

    Args:
        responses (str): The response from the evaluation/annotation model
        aspect (str): The aspect for which the response was generated
    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the ratings and rationales for each completion
    """
    # Split completions/responses from the individual models
    responses = responses.split("\n\n")
    assert len(responses) == 4  # TODO: Make this not hardcoded

    annotation = []
    try:
        if aspect in ["instruction_following", "honesty"]:
            pattern = r"Rating: (.+?)\nRationale: (.+)"
            for response in responses:
                matches = re.search(pattern, response, re.DOTALL)
                annotation.append({
                    "Rating": re.findall(r'\b\d+\b', matches.group(1))[0] if matches.group(1) != "N/A" else "N/A",
                    "Rationale": matches.group(2)
                })
        elif aspect in ["truthfulness", "helpfulness"]:
            pattern = r"Type: (.+?)\nRationale: (.+?)\nRating: (.+?)\nRationale: (.+)"
            for response in responses:
                matches = re.search(pattern, response, re.DOTALL)
                annotation.append({
                    "Type": re.findall(r'\b\d+\b', matches.group(1)) if matches.group(1) != "None" else "None",
                    "Rationale": matches.group(2),
                    "Rating": re.findall(r'\b\d+\b', matches.group(3))[0],
                    "Rationale For Rating": matches.group(4)
                })
    except ValueError as e: # TODO: bug process when the response does not follow the format
        print(responses)
        raise ValueError(e)
    except AttributeError as e:
        print(responses)
        raise AttributeError(e)
    return annotation

def get_eval(sys_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
    """
    Calls the evaluation/annotation model to get the annotations for the completions in the dataset.
    The function excepts corresponding system prompt according to the aspect that is being annotated and the user prompt that contains the completions.

    Args:
        sys_prompt (str): The system prompt for the evaluation/annotation model
        user_prompt (str): The user prompt containing the completions
        max_tokens (int): The maximum number of tokens to generate for the completion
    Returns:
        str: The response from the evaluation/annotation model
    """

    if eval_model_name == "gpt-4":
        for _ in range(max_api_retry):
            try:
                response = openai.ChatCompletion.create(**{
                    "model": "gpt-4",
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0,
                        "max_tokens": max_tokens,
                        "top_p": 0.6,
                        "presence_penalty": 0,
                        "frequency_penalty": 0
                })
                content = response["choices"][0]["message"]["content"]
            except Exception as e:
                print(e)
                time.sleep(1)
            else:
                break
    elif eval_model_name == "llama-3.3-70b-instruct":
        # TODO
        pass
    elif eval_model_name == "debug":
        # ! Only used for debugging purposes and does not actually use any model
        if "instruction_following" in user_prompt or "honesty" in user_prompt:
            content = "Rating: 5\nRationale: THIS IS A DEBUG ANNOTATION\n\n"
            content += "Rating: 4\nRationale: THIS IS A DEBUG ANNOTATION\n\n"
            content += "Rating: 3\nRationale: THIS IS A DEBUG ANNOTATION\n\n"
            content += "Rating: 2\nRationale: THIS IS A DEBUG ANNOTATION"
        else:  # truthfulness or helpfulness
            content = "Type: 1\nRationale: THIS IS A DEBUG ANNOTATION\nRating: 5\nRationale: THIS IS A DEBUG ANNOTATION\n\n"
            content += "Type: 2\nRationale: THIS IS A DEBUG ANNOTATION\nRating: 4\nRationale: THIS IS A DEBUG ANNOTATION\n\n"
            content += "Type: None\nRationale: THIS IS A DEBUG ANNOTATION\nRating: 3\nRationale: THIS IS A DEBUG ANNOTATION\n\n"
            content += "Type: 3\nRationale: THIS IS A DEBUG ANNOTATION\nRating: 2\nRationale: THIS IS A DEBUG ANNOTATION"
    
    return content

def annotate(example: Dict) -> Dict:
    """"
    This function annotates an individual example in the dataset according to all aspects defined in the ASPECTS list.
    It extracts the completions from the example and generates random orderings of the completions to avoid order bias.
    The function then takes care about creating the system and user prompts and sends it to the evaluation/annotation model to annotate.

    Args:
        example (Dict): The example to annotate
    Returns:
        Dict: The annotated example
    """
    completions = [dict({"annotations": {aspect: [] for aspect in ASPECTS}}, **completion)
                    for completion in deepcopy(example["completions"])]

    for aspect in ASPECTS:
        # Get additional world knowledge for truthfulness aspect if provided by the dataset
        if dataset == "truthful_qa":
            world_knowledge = "\n".join(["a subset of correct answers: " + str(example["correct_answers"]), 
                                         "a subset of incorrect_answers: " + str(example["incorrect_answers"])])
        elif dataset == "false_qa":
            world_knowledge = "The question is based on a false promise."
        elif dataset == "flan":
            world_knowledge = example["correct_answers"]
        else:
            world_knowledge = "No additional world knowledge for reference."

        # Generate random ordering of completions to avoid order bias
        random_orders = []
        while len(random_orders) < shuffle_num:
            order = list(range(len(example["completions"])))
            random.shuffle(order)

            if order not in random_orders:
                random_orders.append(order)

        for order in random_orders:        
            # Prepare prompt
            format_input = {"instruction": example["instruction"]}
            format_input.update({f"text_{i+1}": example["completions"][o]["response"] for i, o in enumerate(order)})
            if aspect == "truthfulness" or "helpfulness":  # TODO: Check if this actually needs to be done for helpfulness
                format_input.update({"world_knowledge": world_knowledge})

            # Get evaluation for the sample, retrying if the API call fails
            responses = get_eval(system_prompt, user_prompt=prompt_templates[aspect].format(**format_input)) # TODO: some samples in truthful_qa cannot get annotated when aspect = truthfulness/helpfulness, check if this is a bug
            for i in range(max_api_retry):
                try:
                    responses = process(responses, aspect) # gpt-4 format error
                except Exception as e:
                    if i < max_api_retry - 1:
                        # The response most likely did not follow the expected format, get another response
                        responses = get_eval(system_prompt, user_prompt=prompt_templates[aspect].format(**format_input))
                    else:
                        # If the API call fails after the maximum number of retries, print the error and break
                        print(e)
                        break
                else:
                    # Processing the annotation response was successful, add it to the completions
                    for j in range(len(example["completions"])):
                        completions[j]["annotations"][aspect].append(responses[order.index(j)])
                    break
    
    example["completions"] = completions

    return example
    
def incorporate_annotation_to_completions(example):
    pass

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to download and process (e.g. truthful_qa)")
    parser.add_argument("--max_api_retry", type=int, default=10, help="The number of times to retry the API call if it fails")
    parser.add_argument("--eval_model_name", type=str, default="gpt-4", help="The name of the model to use for evaluation")
    parser.add_argument("--shuffle_num", type=int, default=1, help="The number of times to shuffle the completions")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    max_api_retry = args.max_api_retry
    eval_model_name = args.eval_model_name
    shuffle_num = args.shuffle_num

    # Load dataset (should contain completions)
    # TODO: Switch to Huggingface dataset instead of pandas dataframe? Might cause issues with OpenAI API calls
    with open(os.path.join("../comparison_data_generation", "completion_data", dataset_name + ".json"), "r") as f:
        dataset = json.load(f)
    dataset = pd.DataFrame(dataset)

    # TODO: Check if we can parallelize this using dataset.map here or if API rate limits will cause issues
    dataset_dict = []
    dataset = dataset.to_dict('records')
    for data in tqdm(dataset, total=len(dataset), desc="Annotating"):
        dataset_dict.append(annotate(data))

    # Save dataset
    os.makedirs("annotation", exist_ok=True)
    result_path = os.path.join("annotation", dataset_name + "_annotated.json")
    with open(result_path, "w") as f:
        json.dump([{k: v for k, v in data.items()} for data in dataset_dict[:10]], f, indent=4)