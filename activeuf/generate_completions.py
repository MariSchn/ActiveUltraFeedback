import argparse
import json
import os
import os.path as path

from datasets import load_from_disk
from vllm import SamplingParams

from activeuf.schemas import Completion
from activeuf.configs import *
from activeuf.utils import *

logger = get_logger(__name__)

"""
This script is used to generate completions for one dataset using one model with vLLM.
To generate the completions for all models, this script needs to be run multiple times, once for each model.

Example run command:
    python -m activeuf.generate_completions \
        --dataset_path datasets/allenai/ultrafeedback_binarized_cleaned/test_prefs \
        --model_path HuggingFaceTB/SmolLM2-135M-Instruct
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="The path to the prompts dataset")
    
    parser.add_argument("--model_api", type=str, help="The API of the model to use for completions (e.g. gpt-4)")
    parser.add_argument("--model_path", type=str, help="The path of the model to use for completions (e.g. HuggingFaceTB/SmolLM2-135M-Instruct)")
    
    parser.add_argument("--max_num_gpus", type=int, default=MAX_NUM_GPUS, help="The maximum number of GPUs to use")
    parser.add_argument("--max_tokens", type=int, default=COMPLETION_MAX_TOKENS, help="The maximum number of tokens to generate for each completion")
    parser.add_argument("--temperature", type=int, default=COMPLETION_TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--top_p", type=int, default=COMPLETION_TOP_P, help="top_p value for generation")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for random sampling")

    parser.add_argument("--output_path", type=str, help="Where to save the generated completions")
    args = parser.parse_args()

    if args.output_path is None:

        if args.model_api:
            safe_model_name = args.model_api.replace("/", "_")
        elif args.model_path:
            safe_model_name = args.model_path.replace("/", "_")
        else:
            raise ValueError("Either model_api or model_path must be provided")
        
        args.output_path = os.path.join(
            f"{args.dataset_path.rstrip('/')}-with-{safe_model_name}-completions",
        )
        assert not os.path.exists(args.output_path), f"Output path {args.output_path} already exists"

        logger.info(f"Exporting to {args.output_path} because no output path was provided")

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

    # Load prompts dataset (ensure it follows PromptWithCompletions schema)
    logger.info(f"Loading {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path).map(
        lambda x: PromptWithCompletions(**x).model_dump()
    )

    # Identify samples for which completions with this model need to be generated
    logger.info(f"Size of dataset: {len(dataset)}")
    idxs_needing_completion = []
    for i, sample in enumerate(dataset):
        models_done = set(
            [_["model_api"] for _ in sample["completions"]] + \
            [_["model_path"] for _ in sample["completions"]]
        )
        models_done.discard(None)
        if set([args.model_api, args.model_path]).isdisjoint(models_done):
            idxs_needing_completion.append(i)

    logger.info(f"Found {len(idxs_needing_completion)} samples needing completions")
    if len(idxs_needing_completion) == 0:
        exit(0)

    # Load generation model and prepare sampling params
    if args.model_path:
        logger.info(f"Loading model from {args.model_path} for completion generation")
        model = load_model(args.model_path, max_num_gpus=args.max_num_gpus)
        tokenizer = model.get_tokenizer()
        stop = tokenizer.eos_token if tokenizer and tokenizer.eos_token else None
    else:
        logger.info(f"Using API for {args.model_api} for completion generation")
        model = None  # Can not load the model if it is only called through an API
        stop = None

    sampling_params = SamplingParams(
        max_tokens = args.max_tokens,
        temperature = args.temperature,
        top_p = args.top_p,
        stop = stop,   
    )

    # Construct messages for samples needing completions
    logger.info("Constructing messages for generation model")
    sampled_principles = [
        sample_principle(dataset[i]["source"]) for i in idxs_needing_completion
    ]
    sampled_system_prompts = [
        sample_system_prompt(principle) for principle in sampled_principles
    ]

    all_messages = [
        [
            Message(role="system", content=system_prompt).model_dump(),
            Message(role="user", content=dataset[i]["prompt"]).model_dump(),
        ]
        for system_prompt, i in zip(sampled_system_prompts, idxs_needing_completion)
    ]

    # Generate responses
    logger.info("Generating responses (this may take a while)")
    all_response_texts = get_response_texts(
        all_messages, 
        model = model, 
        model_api = args.model_api, 
        sampling_params = sampling_params,
    )

    logger.info("Formatting responses to follow Completion schema")
    all_completions = []
    for principle, system_prompt, messages, response_text in zip(
        sampled_principles,
        sampled_system_prompts,
        all_messages, 
        all_response_texts,
    ):
        all_completions.append(
            Completion(
                model_api = args.model_api,
                model_path = args.model_path,
                principle = principle,
                system_prompt = system_prompt,
                messages = messages,
                response_text = response_text,
            ).model_dump()
        )

    # Update dataset with completions
    idx2new_completion = dict(zip(idxs_needing_completion, all_completions))
    def add_completion(sample, idx):
        new_completion = idx2new_completion.get(idx)
        if new_completion:
            sample["completions"].append(new_completion)
        return sample
    dataset = dataset.map(add_completion, with_indices=True)

    # Export dataset
    logger.info(f"Exporting dataset to {args.output_path}")
    dataset.save_to_disk(args.output_path)

    # Export args
    args_path = path.join(args.output_path, "args.json")
    with open(args_path, "w") as f_out:
        json.dump(vars(args), f_out)