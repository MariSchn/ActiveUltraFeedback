import argparse
import json
import os.path as path

from datasets import load_from_disk
from vllm import SamplingParams
from transformers import pipeline

from activeuf.schemas import Completion, PromptWithCompletions
from activeuf.configs import *
from activeuf.utils import *

logger = get_logger(__name__)

"""
This script is used to generate completions for one dataset using one model with vLLM.
To generate the completions for all models, this script needs to be run multiple times, once for each model.

Example run command:
    python -m activeuf.generate_completions \
        --dataset_path datasets/allenai/ultrafeedback_binarized_cleaned/test_prefs \
        --model_name HuggingFaceTB/SmolLM2-135M-Instruct \
        --model_class transformers
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="The path to the prompts dataset")
    parser.add_argument("--model_name", type=str, required=True, help="The Huggingface path or API of the model to use for completions (e.g. HuggingFaceTB/SmolLM2-135M-Instruct, gpt-4)")
    
    parser.add_argument("--model_class", type=str, help="How the HuggingFace model for completions should be loaded", choices=["transformers", "pipeline", "vllm"], default=DEFAULT_MODEL_CLASS)

    parser.add_argument("--max_num_gpus", type=int, default=MAX_NUM_GPUS, help="The maximum number of GPUs to use")
    parser.add_argument("--max_tokens", type=int, default=COMPLETION_MAX_TOKENS, help="The maximum number of tokens to generate for each completion")
    parser.add_argument("--temperature", type=int, default=COMPLETION_TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--top_p", type=int, default=COMPLETION_TOP_P, help="top_p value for generation")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for random sampling")

    parser.add_argument("--output_path", type=str, help="Where to save the generated completions")

    parser.add_argument("--debug", action="store_true", help="If set, will only generate completions for the first 10 samples")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = path.join(
            f"{args.dataset_path.rstrip('/')}-with-completions", 
            args.model_name.replace("/", "_"),
        )
        assert not path.exists(args.output_path), f"Output path {args.output_path} already exists"

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
    if args.debug:
        logger.info("Debug mode: only generating completions for the first 10 samples")
        dataset = dataset.select(range(10))

    # Identify samples for which completions with this model need to be generated
    logger.info(f"Size of dataset: {len(dataset)}")
    idxs_needing_completion = []
    for i, sample in enumerate(dataset):
        models_done = {_["model"] for _ in sample["completions"]}
        if args.model_name not in models_done:
            idxs_needing_completion.append(i)

    logger.info(f"Found {len(idxs_needing_completion)} samples needing completions")
    if len(idxs_needing_completion) == 0:
        exit(0)

    # Load generation model and tokenizer, and prepare sampling params
    logger.info(f"Using {args.model_name} for completion generation")
    model, tokenizer = load_model(args.model_name, args.model_class, args.max_num_gpus)
    model.eval()
    sampling_params = SamplingParams(
        max_tokens = args.max_tokens,
        temperature = args.temperature,
        top_p = args.top_p,
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
    with torch.inference_mode():
        all_response_texts = get_response_texts(
            all_messages = all_messages, 
            model = model,
            tokenizer = tokenizer, 
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
                model = args.model_name,
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