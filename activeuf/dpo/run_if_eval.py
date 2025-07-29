import argparse
import json
import os
import subprocess

import vllm

from activeuf.utils import *
from activeuf.configs import *


logger = get_logger(__name__)


"""
This script evaluates a model (either from Hugging Face or a local path) on the IFEval benchmark.
It loads the model, generates a response for every prompt in the benchmark, and runs the evaluation.
The evaluation code is taken from the original IFEval repository https://github.com/google-research/google-research/tree/master/instruction_following_eval

Example usage:
    python -m activeuf.dpo.run_if_eval --model_path allenai/Llama-3.1-Tulu-3-8B-SFT --output_dir ./results
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True, help="The path to the model to evaluate, either Hugging Face or local (e.g. HuggingFaceTB/SmolLM2-135M-Instruct)")
    parser.add_argument("--output_dir", type=str, default="./results", help="The directory to save the evaluation results")
    parser.add_argument("--if_eval_dir", type=str, default="./activeuf/dpo/instruction_following_eval", help="The directory containing the IFEval benchmark code")
    
    parser.add_argument("--model_class", type=str, default=DEFAULT_MODEL_CLASS, help="The model class to use for the evaluation. [transformers, pipeline, vllm]")   
    
    parser.add_argument("--max_num_gpus", type=int, default=MAX_NUM_GPUS, help="The maximum number of GPUs to use")
    parser.add_argument("--max_tokens", type=int, default=COMPLETION_MAX_TOKENS, help="The maximum number of tokens to generate for each completion")
    parser.add_argument("--temperature", type=int, default=COMPLETION_TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--top_p", type=int, default=COMPLETION_TOP_P, help="top_p value for generation")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for random sampling")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    # Login to HF
    logger.info("Logging into HuggingFace")
    setup(login_to_hf=True)

    # Set random seed
    logger.info(f"Setting random seed to {args.seed}")
    if isinstance(args.seed, int):
        set_seed(args.seed)

    # Load the dataset
    dataset_path = f"{args.if_eval_dir}/data/input_data.jsonl"
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, "r") as f:
        prompts = [json.loads(line) for line in f]

    # Load the model
    logger.info(f"Loading model {args.model_path}")
    model, tokenizer = load_model(args.model_path, model_class=args.model_class, max_num_gpus=args.max_num_gpus)
    sampling_params = vllm.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Setup messages to be used by `get_response_texts`
    messages = [
        [
            {
                "role": "user",
                "content": prompt["prompt"]
            }
        ]
        for prompt in prompts
    ]

    # Generate completions
    logger.info("Generating responses for each prompt")
    responses, _ = get_response_texts(
        model=model,
        tokenizer=tokenizer,
        all_messages=messages,
        sampling_params=sampling_params,
    )

    # Save the responses
    logger.info(f"Saving responses to {args.output_dir}/responses.jsonl")
    output_path = f"{args.output_dir}/responses.jsonl"

    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for prompt, response in zip(prompts, responses):
            f.write(json.dumps({
                "prompt": prompt["prompt"],
                "response": response,
            }) + "\n")

    # Run the evaluation
    logger.info("Running IFEval evaluation")

    # Download the necessary NLTK data files
    import nltk
    nltk.download('punkt_tab')

    input_data = os.path.abspath(os.path.join(args.if_eval_dir, "data/input_data.jsonl"))
    input_response_data = os.path.abspath(os.path.join(args.output_dir, "responses.jsonl"))
    output_path = os.path.abspath(args.output_dir)

    os.chdir(os.path.dirname(args.if_eval_dir))

    cmd = [
        "python3", "-m", 
        "instruction_following_eval.evaluation_main",
        f"--input_data={input_data}",
        f"--input_response_data={input_response_data}",
        f"--output_dir={output_path}",
    ]

    with open(os.path.join(output_path, "output.txt"), "w") as outfile:
        subprocess.run(cmd, check=True, stdout=outfile, stderr=subprocess.STDOUT)
    exit()
