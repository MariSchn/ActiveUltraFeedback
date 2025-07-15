import argparse
import json
import os.path as path

from datasets import Dataset, load_from_disk
from vllm import SamplingParams

from activeuf.configs import *
from activeuf.schemas import *
from activeuf.utils import *
from activeuf.prompts_v1 import *

ASPECT2ANNOTATION_PROMPT = {
    "instruction_following": INSTRUCTION_FOLLOWING_ANNOTATION_SYSTEM_PROMPT,
    "honesty": HONESTY_ANNOTATION_SYSTEM_PROMPT,
    "truthfulness": TRUTHFULNESS_ANNOTATION_SYSTEM_PROMPT,
    "helpfulness": HELPFULNESS_ANNOTATION_SYSTEM_PROMPT,
}

logger = get_logger(__name__)

"""
This script is used to annotate the completions generated from the generate_completions.py script.
It uses a LLM as a judge to rate the completions based on the aspects defined in the configs.py file and provides critique/feedback for each completion.

Example run command:
    python -m activeuf.get_raw_annotations \
        --dataset_path datasets/merged_completions \
        --model_name meta-llama/Llama-3.3-70B-Instruct \
        --max_tokens 24000 \
        --download_dir ./hf_cache \
        --output_path datasets/raw_annotations2 \
        --debug
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="The path to the dataset with completions to be annotated")
    parser.add_argument("--model_name", type=str, required=True, help="The Huggingface path or API of the model to use for completions (e.g. HuggingFaceTB/SmolLM2-135M-Instruct, gpt-4)")

    parser.add_argument("--seed", type=int, default=SEED, help="Seed for random sampling")
    parser.add_argument("--max_num_gpus", type=int, default=MAX_NUM_GPUS, help="The maximum number of GPUs to use")
    parser.add_argument("--model_class", type=str, default=DEFAULT_MODEL_CLASS, help="The class which is used to perform inference (e.g. transformers, pipeline, vllm)")
    
    parser.add_argument("--max_tokens", type=int, default=ANNOTATION_MAX_TOKENS, help="The maximum number of tokens for LLM responses")
    parser.add_argument("--temperature", type=float, default=ANNOTATION_TEMPERATURE, help="The temperature for sampling")
    parser.add_argument("--top_p", type=float, default=ANNOTATION_TOP_P, help="The top_p for sampling")

    parser.add_argument("--download_dir", type=str, help="The path to the Huggingface cache directory. If not set, the default Huggingface cache directory is used.")
    parser.add_argument("--output_path", type=str, help="Where to export the annotated dataset")
    parser.add_argument("--debug", action="store_true", help="If set, will only annotate the first few samples")
    args = parser.parse_args()

    if not args.output_path:
        args.output_path = f"datasets/raw_annotations"
    assert not path.exists(args.output_path), f"Output path {args.output_path} already exists"

    return args

if __name__ == "__main__":
    args = parse_args()

    logger.info("Logging into HuggingFace")
    setup(login_to_hf=True)

    if args.seed:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    logger.info(f"Loading {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    if args.debug:
        logger.info("Debug mode: only annotating completions for the first few prompts")
        dataset = dataset.select(range(50))
    logger.info(f"{len(dataset)}")

    logger.info(f"Using {args.model_name} for annotation")
    if args.download_dir:
        model, tokenizer = load_model(args.model_name, args.model_class, download_dir=args.download_dir)
    else:
        model, tokenizer = load_model(args.model_name, args.model_class)
    sampling_params = SamplingParams(
        max_tokens = args.max_tokens,
        temperature = args.temperature,
        top_p = args.top_p,      
    )

    logger.info("Accruing messages and metadata for annotation")
    all_messages = []
    all_metadata = []
    for prompt in tqdm(dataset):
        for aspect, annotation_prompt in ASPECT2ANNOTATION_PROMPT.items():
            messages = [
                {
                    "role": "system",
                    "content": PREFERENCE_ANNOTATION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": "### Input\n\n" + \
                        f"Instruction: {annotation_prompt}\n\n" + \
                        "\n\n".join(f"{_['model']}: {_['response_text']}" for _ in prompt["completions"]) + \
                        "### Output\n\n",
                }
            ]
            all_messages.append(messages)
            all_metadata.append({
                "prompt_id": prompt["prompt_id"],
                "aspect": aspect,
            })
        messages = [
            {
                "role": "system",
                "content": FEEDBACK_ANNOTATION_SYSTEM_PROMPT ,
            },
            {
                "role": "user",
                "content": "### Input\n\n" + \
                        f"Instruction: {annotation_prompt}\n\n" + \
                        "\n\n".join(f"{_['model']}: {_['response_text']}" for _ in prompt["completions"]) + \
                        "### Output\n\n",
            }
        ]
        all_messages.append(messages)
        all_metadata.append({
            "prompt_id": prompt["prompt_id"],
            "aspect": "critique",
        })
    logger.info(f"Number of annotation messages: {len(all_messages)}")

    logger.info("Annotating completions")
    all_raw_annotations = get_response_texts(model, tokenizer, all_messages, sampling_params)

    logger.info(f"Saving raw annotations to {args.output_path}")
    for metadata, messages, raw_annotations in zip(all_metadata, all_messages, all_raw_annotations):
        metadata["messages"] = messages
        metadata["raw_annotations"] = raw_annotations
    Dataset.from_list(all_metadata).save_to_disk(args.output_path)

    args_path = path.join(args.output_path, "args.json")
    with open(args_path, "w") as f_out:
        json.dump(vars(args), f_out)