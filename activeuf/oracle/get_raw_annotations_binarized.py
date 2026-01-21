import argparse
import json
import os.path as path
import numpy as np
from tqdm import tqdm

from datasets import Dataset, load_from_disk, load_dataset
from vllm import SamplingParams
from transformers import AutoTokenizer

from activeuf.utils import get_logger, set_seed, setup, load_model, get_response_texts
from activeuf.oracle.prompts import (
    PREFERENCE_ANNOTATION_SYSTEM_PROMPT,
    HELPFULNESS_ANNOTATION_SYSTEM_PROMPT,
    HONESTY_ANNOTATION_SYSTEM_PROMPT,
    TRUTHFULNESS_ANNOTATION_SYSTEM_PROMPT,
    INSTRUCTION_FOLLOWING_ANNOTATION_SYSTEM_PROMPT,
)
import os

ASPECT2ANNOTATION_PROMPT = {
    "instruction_following": INSTRUCTION_FOLLOWING_ANNOTATION_SYSTEM_PROMPT,
    "honesty": HONESTY_ANNOTATION_SYSTEM_PROMPT,
    "truthfulness": TRUTHFULNESS_ANNOTATION_SYSTEM_PROMPT,
    "helpfulness": HELPFULNESS_ANNOTATION_SYSTEM_PROMPT,
}

logger = get_logger(__name__)

"""
This script is used to annotate the allenai/ultrafeedback_binarized_cleaned dataset.
It annotates BOTH the chosen and rejected responses for each prompt.

Example run command:
    python -u -m activeuf.oracle.get_raw_annotations_binarized \
        --dataset_path allenai/ultrafeedback_binarized_cleaned \
        --model_name="Qwen/Qwen3-235B-A22B" \
        --max_tokens 24000 \
        --output_path /iopsstor/scratch/cscs/dmelikidze/datasets/ultrafeedback_binarized_annotated/ \
        --model_class vllm_server \
        --temperature 0.0 \
        --top_p 0.1 \
        --num_nodes 2 \
        --batch_size 1000
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--batch_start",
            type=int,
            default=None,
            help="Start index (inclusive) of the batch to annotate. If not set, starts from 0.",
        )
    parser.add_argument(
            "--batch_end",
            type=int,
            default=None,
            help="End index (exclusive) of the batch to annotate. If not set, goes to end.",
        )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="allenai/ultrafeedback_binarized_cleaned",
        help="The HuggingFace dataset path (default: allenai/ultrafeedback_binarized_cleaned)",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train_prefs",
        help="The dataset split to use (default: train_prefs)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The Huggingface path of the model to use for annotation (e.g. Qwen/Qwen3-235B-A22B)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random sampling")
    parser.add_argument(
        "--max_num_gpus",
        type=int,
        default=4,
        help="The maximum number of GPUs to use",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="The number of nodes to use",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="vllm_server",
        help="The class which is used to perform inference (e.g. transformers, pipeline, vllm, vllm_server)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=24000,
        help="The maximum number of tokens for LLM responses",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling",
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.1, 
        help="The top_p for sampling"
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        help="The path to the Huggingface cache directory.",
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="Where to export the annotated dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="The number of rows to annotate in one batch",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, will only annotate the first few samples",
    )
    parser.add_argument(
        "--enable_reasoning",
        action="store_true",
        help="If set, enables chain-of-thought reasoning for Qwen3 models",
    )
    parser.add_argument(
        "--reasoning_max_tokens",
        type=int,
        default=8192,
        help="Max tokens for reasoning generation (used when --enable_reasoning is set)",
    )

    # Direct output mode: save LLM output directly to annotation columns
    # Removed direct_output and direct_output_field arguments

    args = parser.parse_args()
    return args


## Removed broken/unused calculate_probabilities_openai definition
def get_thinking_suffix_token_count(tokenizer):
    """
    Get the exact number of tokens that comprise '</think>\n\n' for this tokenizer.
    This is the suffix after reasoning before actual content in Qwen3 models.
    """
    # The exact string that appears at the end of thinking block
    thinking_suffix = "</think>\n\n"
    tokens = tokenizer.encode(thinking_suffix, add_special_tokens=False)
    return len(tokens), tokens


def find_thinking_suffix_position(content_logprobs, suffix_tokens, tokenizer):
    """
    Find the position where the thinking suffix starts in the logprobs content.
    Returns the index where '</think>\n\n' starts, or -1 if not found.
    """
    suffix_len = len(suffix_tokens)
    
    # Convert suffix token ids to their string representations for matching
    suffix_token_strs = [tokenizer.decode([t]) for t in suffix_tokens]
    
    # Scan through looking for the suffix sequence
    for i in range(len(content_logprobs) - suffix_len + 1):
        match = True
        for j in range(suffix_len):
            if content_logprobs[i + j].token != suffix_token_strs[j]:
                match = False
                break
        if match:
            return i
    
    return -1


def calculate_probabilities_after_thinking(raw_output, tokenizer, target_words=["1", "2", "3", "4", "5"]):
    """
    Calculate probabilities for target words from API outputs,
    taking logprobs from the first token after the '</think>\n\n' suffix.
    Returns None for samples where thinking didn't complete properly.
    """
    suffix_token_count, suffix_tokens = get_thinking_suffix_token_count(tokenizer)
    word_probabilities = []
    for output in raw_output:
        content_logprobs = output.choices[0].logprobs.content
        suffix_start_pos = find_thinking_suffix_position(content_logprobs, suffix_tokens, tokenizer)
        if suffix_start_pos == -1:
            logger.warning(f"Thinking suffix '</think>\\n\\n' not found in output - reasoning incomplete")
            word_probabilities.append(None)
            continue
        first_content_pos = suffix_start_pos + suffix_token_count
        if first_content_pos >= len(content_logprobs):
            logger.warning(f"No token after thinking suffix - generation ended prematurely")
            word_probabilities.append(None)
            continue
        top_logprobs = content_logprobs[first_content_pos].top_logprobs
        token_logprobs = {token_logprob.token: token_logprob.logprob for token_logprob in top_logprobs}
        target_logprobs = {word: token_logprobs.get(word, -float("inf")) for word in target_words}
        exp_values = [np.exp(lp) for lp in target_logprobs.values()]
        total = sum(exp_values)
        if total == 0:
            prob_dict = {k: 0.0 for k in target_logprobs.keys()}
        else:
            prob_dict = {k: float(v) / total for k, v in zip(target_logprobs.keys(), exp_values)}
        word_probabilities.append(prob_dict)
    return word_probabilities


def extract_response_text(messages):
    """
    Extract the response text from the chosen/rejected message format.
    The format is a list of dicts with 'role' and 'content' keys.
    We want the assistant's response (last message with role='assistant').
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def extract_prompt_text(messages):
    """
    Extract the prompt/instruction text from the message format.
    We want the user's message (first message with role='user').
    """
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def load_dataset_with_resume(dataset_path, dataset_split, output_path):
    """
    Load the dataset and handle resumption if output already exists.
    """
    logger.info(f"Loading dataset from {dataset_path}, split: {dataset_split}")
    dataset = load_dataset(dataset_path, split=dataset_split)
    
    already_processed_rows = 0
    already_processed_dataset = None
    
    if os.path.exists(output_path):
        logger.info(f"Output path {output_path} exists. Checking for resume...")
        try:
            already_processed_dataset = load_from_disk(output_path)
            already_processed_rows = len(already_processed_dataset)
            logger.info(f"Found {already_processed_rows} already processed rows.")
            
            if already_processed_rows >= len(dataset):
                logger.info("All rows already processed. Exiting.")
                exit(0)
                
            dataset = dataset.select(range(already_processed_rows, len(dataset)))
            logger.info(f"Resuming from row {already_processed_rows}. {len(dataset)} rows remaining.")
            
        except Exception as e:
            logger.warning(f"Could not load existing output: {e}. Starting fresh.")
            already_processed_dataset = None
    
    return dataset, already_processed_dataset


if __name__ == "__main__":
    args = parse_args()

    print("=== Arguments ===")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    logger.info("Logging into HuggingFace")
    setup(login_to_hf=True)

    if args.seed:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)


    dataset, already_processed_dataset = load_dataset_with_resume(
        args.dataset_path, args.dataset_split, args.output_path
    )

    # Apply batch interval slicing if specified
    batch_start = args.batch_start if args.batch_start is not None else 0
    batch_end = args.batch_end if args.batch_end is not None else len(dataset)
    dataset = dataset.select(range(batch_start, min(batch_end, len(dataset))))

    # Initialize output list from already processed data
    if already_processed_dataset is not None:
        output_dataset = already_processed_dataset.to_list()
    else:
        output_dataset = []

    if args.debug:
        logger.info("Debug mode: only annotating first 50 samples")
        dataset = dataset.select(range(min(50, len(dataset))))

    logger.info(f"Dataset size: {len(dataset)}")

    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("HF_CACHE:", os.environ.get("HF_CACHE"))

    logger.info(f"Loading model: {args.model_name}")
    model, _ = load_model(
        model_name=args.model_name,
        model_class=args.model_class,
        max_num_gpus=args.max_num_gpus,
        num_nodes=args.num_nodes,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Use higher max_tokens when reasoning is enabled
    generation_max_tokens = args.reasoning_max_tokens if args.enable_reasoning else 64
    
    sampling_params = SamplingParams(
        max_tokens=generation_max_tokens,
        temperature=float(args.temperature) if not args.enable_reasoning else 0.6,  # Slightly higher temp for reasoning
        top_p=float(args.top_p) if not args.enable_reasoning else 0.95,
        logprobs=20,  # Always request logprobs to get distribution after reasoning
    )
    
    # Prepare generate_kwargs for enabling thinking
    generate_kwargs = {"enable_thinking": args.enable_reasoning}

    logger.info("Starting annotation process...")
    
    batch_size = args.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    os.makedirs(args.output_path, exist_ok=True)
    args_path = path.join(args.output_path, "args.json")
    with open(args_path, "w") as f_out:
        json.dump(vars(args), f_out, indent=2)

    n_aspects = len(ASPECT2ANNOTATION_PROMPT.keys())
    aspects = list(ASPECT2ANNOTATION_PROMPT.keys())
    response_types = ["chosen", "rejected"]
    # Removed direct_output print
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(dataset))
        batch_dataset = dataset.select(range(start, end))

        # Build messages for both chosen and rejected responses
        batch_messages = []
        batch_metadata = []
        
        for row_idx, row in enumerate(batch_dataset):
            # Get prompt - either from 'prompt' field or extract from messages
            if "prompt" in row and row["prompt"]:
                prompt_text = row["prompt"]
            else:
                # Extract from chosen messages
                prompt_text = extract_prompt_text(row["chosen"])
            
            # Process both chosen and rejected responses
            for response_type in response_types:
                response_messages = row[response_type]
                response_text = extract_response_text(response_messages)
                
                # Create annotation messages for each aspect
                for aspect, annotation_prompt in ASPECT2ANNOTATION_PROMPT.items():
                    messages = [
                        {
                            "role": "system",
                            "content": PREFERENCE_ANNOTATION_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": annotation_prompt.format(
                                prompt=prompt_text,
                                completion=response_text,
                            ),
                        },
                    ]
                    batch_messages.append(messages)
                    batch_metadata.append({
                        "row_idx": start + row_idx,
                        "response_type": response_type,
                        "aspect": aspect,
                    })

        # Get model responses
        all_response_texts, all_raw_objects = get_response_texts(
            model, tokenizer, batch_messages, sampling_params, generate_kwargs=generate_kwargs
        )


        # Always calculate probabilities, ignore direct_output
        # Always extract logprobs at the first token after '</think>\n\n' suffix
        all_probabilities = calculate_probabilities_after_thinking(
            all_raw_objects, tokenizer, target_words=["1", "2", "3", "4", "5"]
        )


        # Organize results by row
        # Each row has 2 response types * n_aspects annotations
        annotations_per_row = len(response_types) * n_aspects

        for row_local_idx in range(len(batch_dataset)):
            row = batch_dataset[row_local_idx]
            global_row_idx = start + row_local_idx

            # Get prompt
            if "prompt" in row and row["prompt"]:
                prompt_text = row["prompt"]
            else:
                prompt_text = extract_prompt_text(row["chosen"])

            # Build output row
            output_row = {
                "row_idx": global_row_idx,
                "prompt": prompt_text,
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "chosen_annotations": {},
                "rejected_annotations": {},
            }

            # Copy any additional fields from original dataset
            for key in row.keys():
                if key not in ["prompt", "chosen", "rejected"]:
                    output_row[key] = row[key]

            # Extract annotations for this row
            base_idx = row_local_idx * annotations_per_row

            # Save probability dicts as before (always)
            for j, aspect in enumerate(aspects):
                output_row["chosen_annotations"][aspect] = all_probabilities[base_idx + j]
            for j, aspect in enumerate(aspects):
                output_row["rejected_annotations"][aspect] = all_probabilities[base_idx + n_aspects + j]

            output_dataset.append(output_row)

        # Save after each batch
        Dataset.from_list(output_dataset).save_to_disk(args.output_path)
        logger.info(f"Saved batch {batch_idx + 1}/{num_batches} ({len(output_dataset)} total rows)")

    logger.info(f"Annotation complete! Total rows: {len(output_dataset)}")
    logger.info(f"Output saved to: {args.output_path}")
