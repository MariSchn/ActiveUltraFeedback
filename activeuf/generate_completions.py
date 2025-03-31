import argparse
from dotenv import load_dotenv
import json
import os
import os.path as path
import random
from tqdm import tqdm
from typing import Generator

import torch
from vllm import SamplingParams

from activeuf.schemas import Sample, Completion
from activeuf.configs import DATASET_POOL, MAX_NUM_GPUS, PRINCIPLE2PROMPTS
from activeuf.utils import (
    set_seed, sample_principle_for_dataset, get_stop_tokens,
    load_model,
)

from activeuf.comparison_data_generation.fastchat import conv_template

load_dotenv()

# TODO: update these instructions
"""
This script is used to generate completions for one dataset using one model with vLLM.
To generate the completions for all models, this script needs to be run multiple times, once for each model.
Before this script is run, the dataset need to be created using the `create_raw_dataset.py` script.

To run it, you need to pass the model (`model_name`) and the dataset (`dataset`) as arguments. 
The `model_name` is the name of the model to use for completions (e.g. llama-2-13b-chat) and the `dataset` is the name of the dataset to download and process (e.g. truthful_qa).
To map the inputs to the actual model path or Huggingface model the `model_path` dictionary is used and needs to be updated if you want to use a new model.
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_path", type=str, required=True, help="The path to the input dataset")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use for completions (e.g. llama-2-13b-chat)")

    parser.add_argument("--max_num_gpus", type=int, default=MAX_NUM_GPUS, help="The maximum number of GPUs to use")
    parser.add_argument("--max_tokens", type=int, default=1024, help="The maximum number of tokens to generate for each completion")
    parser.add_argument("--seed", type=int, default=123, help="Seed for random sampling")

    parser.add_argument("--temperature", type=int, default=1, help="Temperature for generation")
    parser.add_argument("--top_p", type=int, default=1, help="top_p value for generation")

    parser.add_argument("--output_dir", type=str, default="datasets/datasets_with_completions/", help="The directory for exporting the generated completions")
    return parser.parse_args()

def load_input_data(input_dataset_path: str) -> Generator[Sample, None, None]:
    with open(input_dataset_path, "r") as f_in:
        for line in tqdm(f_in):
            yield Sample(**json.loads(line))

def prepare_prompt(
        sample: Sample, 
        model_name: str, 
        principle_prompt: str,
    ) -> str:
    # prepare conv template
    # TODO: Make nicer
    # TODO: Adapt for more models (currently just gpt-2 is tested)
    if "ultralm" in model_name:
        system_prompt = "User: A one-turn chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, very detailed, and polite answers to the user's questions.</s>"
        system_prompt += "User: " + principle_prompt + "</s>"
        conv = [system_prompt]
        conv.append("User: " + sample["instruction"] + "</s>")
        conv.append("Assistant: ")
        prompt = "\n".join(conv)
    elif "starchat" in model_name:
        system_prompt = "<|system|>" + principle_prompt + "<|end|>"
        conv = [system_prompt]
        conv.append("<|user|>" + sample["instruction"] + "<|end|>")
        conv.append("<|assistant|>")
        prompt = "\n".join(conv)
    elif model_name == "wizardlm-7b":
        prompt = "{}\n\n### Response:".format(sample["instruction"])
    elif model_name.split("-")[0] in ["llama", "alpaca", "vicuna", "mpt", "falcon", "wizardlm"]: # note that the wizardlm should be 13b or 30b
        conv = conv_template[model_name.split("-")[0]].copy()
        conv.system += " " + principle_prompt
        conv.append_message(conv.roles[0], sample["instruction"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif model_name == "gpt-2":  # Only used for testing, as gpt-2 is a relatively small model that can be easily loaded
        prompt = sample["instruction"]
    else:
        raise NotImplementedError

    return prompt

if __name__ == "__main__":
    args = parse_args()

    # extract dataset name from path
    dataset_name = path.splitext(path.basename(args.input_dataset_path))[0]
    assert dataset_name in DATASET_POOL

    # Set random seed
    if isinstance(args.seed, int):
        set_seed(args.seed)

    # Load generation model
    model = load_model(args.model_name, max_num_gpus=args.max_num_gpus)
    sampling_params = SamplingParams(
        max_tokens = args.max_tokens,
        temperature = args.temperature,
        top_p = args.top_p,
        stop = get_stop_tokens(args.model_name, model),        
    )

    # prepare output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = path.join(args.output_dir, f"{dataset_name}.jsonl")
    f_out = open(output_path, "w")

    # For each sample
    for sample in load_input_data(args.input_dataset_path):
        # perform generation only if model is specified for this sample and completion is not already present
        if args.model_name in sample.model_names and \
            args.model_name not in [_.model_name for _ in sample.completions]:

            # sample guiding principle for generation
            principle = sample_principle_for_dataset(dataset_name)
            principle_prompt = random.choice(PRINCIPLE2PROMPTS[principle])

            # get generation prompt, then generate, then clean the response
            # TODO: simplify this response cleaning
            prompt = prepare_prompt(sample, args.model_name, principle_prompt)
            with torch.inference_mode:
                response = model.generate(prompt, sampling_params)[0]
            response_text = response.outputs[0].text.strip().rstrip("</s>").strip()

            # append completion to sample
            sample.completions.append(Completion(
                model_name = args.model_name,
                principle = principle, 
                principle_prompt = principle_prompt,
                response_text = response_text,
            ))
        
        # export sample
        print(sample.model_dump_json(), file=f_out, flush=True)
        break

    f_out.close()