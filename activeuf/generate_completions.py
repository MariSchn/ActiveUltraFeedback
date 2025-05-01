import argparse
import os
import os.path as path
import random
from tqdm import tqdm

import torch
from vllm import SamplingParams
from transformers import pipeline

from activeuf.schemas import Completion
from activeuf.configs import *
from activeuf.utils import *

# TODO: update these instructions
"""
This script is used to generate completions for one dataset using one model with vLLM.
To generate the completions for all models, this script needs to be run multiple times, once for each model.

Example run command:
    python -m activeuf.generate_completions --input_dataset_path datasets/input_datasets/truthful_qa.jsonl --model_name gpt-2
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_path", type=str, required=True, help="The path to the input dataset")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use for completions (e.g. llama-2-13b-chat)")

    parser.add_argument("--max_num_gpus", type=int, default=MAX_NUM_GPUS, help="The maximum number of GPUs to use")
    parser.add_argument("--max_tokens", type=int, default=COMPLETION_MAX_TOKENS, help="The maximum number of tokens to generate for each completion")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for random sampling")
    parser.add_argument("--model_class", type=str, default=MODEL_CLASS, help="The class which is used to perform inference (e.g. transformers, pipeline, vllm)")

    parser.add_argument("--temperature", type=int, default=COMPLETION_TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--top_p", type=int, default=COMPLETION_TOP_P, help="top_p value for generation")

    parser.add_argument("--output_dir", type=str, default="datasets/datasets_with_completions/", help="The directory for exporting the generated completions")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    setup(login_to_hf=True)

    # extract dataset name from path
    dataset_name = path.splitext(path.basename(args.input_dataset_path))[0]
    assert dataset_name in DATASET_POOL

    # Set random seed
    if isinstance(args.seed, int):
        set_seed(args.seed)

    # Load and filter samples
    samples = load_samples(args.input_dataset_path)
    to_complete = [sample for sample in samples if 
                   args.model_name in sample.model_names and                           # model is specified for this sample
                   args.model_name not in [_.model_name for _ in sample.completions]]  # completion is not already present
    prompts = [sample.instruction for sample in to_complete]
    completion_objects = []

    if not to_complete:
        print(f"Model {args.model_name} does not have any samples to complete. Exiting...")
        exit(0)

    # Load generation model
    model, tokenizer = load_model(args.model_name, args.max_num_gpus, args.model_class)
    sampling_params = SamplingParams(
        max_tokens = args.max_tokens,
        temperature = args.temperature,
        top_p = args.top_p,      
    )
    model.eval()

    # Sample principles (and subsequently system prompts) for generation
    system_prompts = []
    for sample in to_complete:
        # sample guiding principle for generation
        principle = sample_principle_for_dataset(dataset_name)
        principle_prompt = random.choice(PRINCIPLE2PROMPTS[principle])

        system_prompts.append(principle_prompt)

        completion_objects.append(Completion(
            model_name = args.model_name,
            principle = principle, 
            principle_prompt = principle_prompt,
            response_text = "",
        ))

    # Generate completions
    with torch.inference_mode():
        completion_texts = get_completion(prompts, model, tokenizer, args.model_name, sampling_params, system_prompts)

    # Add completions to samples
    for sample, completion_object, completion_text in zip(to_complete, completion_objects, completion_texts):
        completion_object.response_text = completion_text
        sample.completions.append(completion_object)

    # Export samples
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{dataset_name}.jsonl")
    f_out = open(out_path, "w")
    for sample in samples:
        print(sample.model_dump_json(), file=f_out, flush=True)
    f_out.close()

    # Free up memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
