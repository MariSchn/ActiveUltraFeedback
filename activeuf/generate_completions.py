import argparse
import os
import os.path as path
import random
from tqdm import tqdm

import torch
from vllm import SamplingParams

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
    parser.add_argument("--max_tokens", type=int, default=1024, help="The maximum number of tokens to generate for each completion")
    parser.add_argument("--seed", type=int, default=123, help="Seed for random sampling")

    parser.add_argument("--temperature", type=int, default=1, help="Temperature for generation")
    parser.add_argument("--top_p", type=int, default=1, help="top_p value for generation")

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

    # Load generation model
    model = load_model(args.model_name, max_num_gpus=args.max_num_gpus)
    sampling_params = SamplingParams(
        max_tokens = args.max_tokens,
        temperature = args.temperature,
        top_p = args.top_p,
        stop = get_stop_tokens(args.model_name, model),        
    )

    # Prepare output file
    # Use a temporary file to avoid overwriting in the case of the output file also being the input file
    # TODO: Make completion generation parallelizable
    os.makedirs(args.output_dir, exist_ok=True)
    temp_output_path = path.join(args.output_dir, f"{dataset_name}_{args.model_name}_temp.jsonl")
    final_output_path = path.join(args.output_dir, f"{dataset_name}.jsonl")
    f_out = open(temp_output_path, "w")

    # For each sample
    for sample in tqdm(yield_samples(args.input_dataset_path)):
        # perform generation only if model is specified for this sample and completion is not already present
        if args.model_name in sample.model_names and \
            args.model_name not in [_.model_name for _ in sample.completions]:

            # sample guiding principle for generation
            principle = sample_principle_for_dataset(dataset_name)
            principle_prompt = random.choice(PRINCIPLE2PROMPTS[principle])

            # construct prompt for generation
            messages = [
                {"role": "system", "content": principle_prompt},
                {"role": "user", "content": sample.instruction},
            ]
            prompt = model.get_tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            # generate, then clean the response
            # TODO: simplify this response cleaning
            with torch.inference_mode():
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

    f_out.close()

    # Overwrite the final output file with the temp output file
    os.replace(temp_output_path, final_output_path)