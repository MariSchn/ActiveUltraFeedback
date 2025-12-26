import argparse
import json
import os
import os.path as path

import torch
import vllm
from datasets import load_from_disk

from activeuf.schemas import Completion, PromptWithCompletions, Message
from activeuf.utils import (
    load_model,
    get_logger,
    set_seed,
    setup,
    get_response_texts,
    sample_principle,
    sample_system_prompt,
)

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
    # fmt: off

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="The path to the prompts dataset")
    parser.add_argument("--model_name", type=str, required=True, help="The Huggingface path or API of the model to use for completions (e.g. HuggingFaceTB/SmolLM2-135M-Instruct, gpt-4)")

    parser.add_argument("--model_class", type=str, help="How the HuggingFace model for completions should be loaded", choices=["transformers", "pipeline", "vllm", "vllm_server"], default="vllm")

    parser.add_argument("--max_num_gpus", type=int, default=4, help="The maximum number of GPUs to use")
    parser.add_argument("--num_nodes", type=int, default=os.getenv("SLURM_NNODES", 1), help="The maximum number of nodes to use for distributed training (if applicable)")
    parser.add_argument("--data_parallel_size", type=int, default=1, help="The size of the data parallel group (only applicable for vllm_server model class)")
    parser.add_argument("--max_tokens", type=int, default=4096, help="The maximum number of tokens to generate for each completion")
    parser.add_argument("--max_model_len", type=int, default=0, help="The maximum context length of the model")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="The GPU memory utilization to use for loading the model (only used for vllm models)")
    parser.add_argument("--temperature", type=int, default=1.0, help="Temperature for generation")
    parser.add_argument("--top_p", type=int, default=1.0, help="top_p value for generation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random sampling")

    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks to split the dataset into")
    parser.add_argument("--chunk_index", type=int, default=0, help="Index of the chunk to process (0-indexed)")

    parser.add_argument("--output_path", type=str, help="Where to save the generated completions")

    parser.add_argument("--debug", action="store_true", help="If set, will only generate completions for the first 10 samples")
    parser.add_argument("--skip_too_long_prompts", action="store_true", help="If set, will skip prompts that exceed the model's max context length instead of failing")
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
    if args.num_chunks > 1:
        dataset = dataset.shard(num_shards=args.num_chunks, index=args.chunk_index)
        print(
            f"Sharding dataset into {args.num_chunks} chunks and selecting chunk {args.chunk_index} (len: {len(dataset)})"
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
    model, tokenizer = load_model(
        model_name=args.model_name,
        model_class=args.model_class,
        max_num_gpus=args.max_num_gpus,
        num_nodes=args.num_nodes,
        data_parallel_size=args.data_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = vllm.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
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

    # Identify prompts that are too long for the model's context length
    valid_indices = set(range(len(all_messages)))
    if args.skip_too_long_prompts:
        max_model_len = model.llm_engine.model_config.max_model_len
        logger.info(f"Checking for prompts longer than {max_model_len} tokens")

        valid_indices = set()
        for i, messages in enumerate(all_messages):
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            num_tokens = len(tokenizer.encode(prompt_text))
            if num_tokens < max_model_len:
                valid_indices.add(i)
            else:
                logger.warning(
                    f"Prompt {idxs_needing_completion[i]} has {num_tokens} tokens (max: {max_model_len}) - will use empty response"
                )

        logger.info(
            f"Found {len(all_messages) - len(valid_indices)} prompts that are too long (will use empty response)"
        )

    # Generate responses only for valid prompts
    logger.info("Generating responses (this may take a while)")
    valid_messages = [all_messages[i] for i in sorted(valid_indices)]

    with torch.inference_mode():
        valid_response_texts, _ = get_response_texts(
            all_messages=valid_messages,
            model=model,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
        )

    # Map responses back to all prompts, using empty string for skipped ones
    all_response_texts = []
    valid_idx_iter = iter(valid_response_texts)
    for i in range(len(all_messages)):
        if i in valid_indices:
            all_response_texts.append(next(valid_idx_iter))
        else:
            all_response_texts.append("")

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
                model=args.model_name,
                principle=principle,
                system_prompt=system_prompt,
                messages=messages,
                response_text=response_text,
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

    # Export first sample
    first_sample = dataset[0]
    first_sample_path = path.join(args.output_path, "first_sample.json")
    with open(first_sample_path, "w") as f_out:
        json.dump(first_sample, f_out)
