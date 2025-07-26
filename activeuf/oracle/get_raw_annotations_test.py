import re
import argparse
import json
import os.path as path
import numpy as np

from datasets import Dataset, load_from_disk, load_dataset
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer

from activeuf.configs import *
from activeuf.schemas import *
from activeuf.utils import *
from activeuf.oracle.prompts import *

# these are not system prompts, these are user prompts.
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
    python -m activeuf.get_raw_annotations_v4b2 \
        --dataset_path allenai/ultrafeedback_binarized_cleaned \
        --model_name="meta-llama/Llama-3.3-70B-Instruct" \
        --max_tokens 24000 \
        --output_path /iopsstor/scratch/cscs/dmelikidze/datasets/combined_annotations_qwen_test10/ \
        --model_class vllm \
        --temperature 0.0 \
        --top_p 0.1 \
        --model_to_annotate "chosen" \
        --batch_size_to_annotate 1000


    python -m activeuf.get_raw_annotations_v4b2 \
        --dataset_path allenai/ultrafeedback_binarized_cleaned \
        --model_name="Qwen/Qwen3-32B" \
        --max_tokens 24000 \
        --output_path /iopsstor/scratch/cscs/dmelikidze/datasets/combined_annotations_qwen_test10/ \
        --model_class vllm \
        --temperature 0.0 \
        --top_p 0.1 \
        --model_to_annotate "chosen" \
        --batch_size_to_annotate 200


    python -m activeuf.get_raw_annotations_v4b2 \
        --dataset_path allenai/ultrafeedback_binarized_cleaned \
        --model_name="meta-llama/Llama-3.3-70B-Instruct" \
        --max_tokens 24000 \
        --output_path /iopsstor/scratch/cscs/dmelikidze/datasets/raw_annotations4 \
        --model_class vllm \
        --temperature 0.0 \
        --top_p 0.1 \
        --debug

"""
# TODO compare qwen's performance to ultrafeedback_binarized dataset. see how many times it chooses the "Chosen" respones.
# 0.571 for llama
# 0.589 now for llama. slightly modified prompt.
# 0.6269 woow... with gpt recommendations.


def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="The path to the dataset with completions to be annotated")
    parser.add_argument("--model_name", type=str, required=True,
                        help="The Huggingface path or API of the model to use for completions (e.g. HuggingFaceTB/SmolLM2-135M-Instruct, gpt-4)")

    parser.add_argument("--seed", type=int, default=SEED,
                        help="Seed for random sampling")
    parser.add_argument("--max_num_gpus", type=int, default=MAX_NUM_GPUS,
                        help="The maximum number of GPUs to use")
    parser.add_argument("--model_class", type=str, default=DEFAULT_MODEL_CLASS,
                        help="The class which is used to perform inference (e.g. transformers, pipeline, vllm)")

    parser.add_argument("--max_tokens", type=int, default=ANNOTATION_MAX_TOKENS,
                        help="The maximum number of tokens for LLM responses")
    parser.add_argument("--temperature", type=float,
                        default=ANNOTATION_TEMPERATURE, help="The temperature for sampling")
    parser.add_argument("--top_p", type=float,
                        default=ANNOTATION_TOP_P, help="The top_p for sampling")

    parser.add_argument("--download_dir", type=str,
                        help="The path to the Huggingface cache directory. If not set, the default Huggingface cache directory is used.")
    parser.add_argument("--output_path", type=str,
                        help="Where to export the annotated dataset")
    parser.add_argument("--debug", action="store_true",
                        help="If set, will only annotate the first few samples")

    parser.add_argument("--model_to_annotate", type=str, required=True,
                        help="The model whose completions to annotate")
    parser.add_argument("--batch_size_to_annotate", type=int, default=50,
                        help="The number of completions to annotate in one batch")

    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.model_to_annotate)

    return args


def calculate_probabilities(raw_output, tokenizer, target_words):
    target_token_ids = [tokenizer.encode(t, add_special_tokens=False)[
        0] for t in target_words]

    word_probabilities = []

    for output in raw_output:
        logprobs = output.outputs[0].logprobs

        logprob_dict = logprobs[0]

        token_logprobs = {}
        for t, tid in zip(target_words, target_token_ids):
            token_logprobs[t] = logprob_dict.get(tid, -float("inf"))

        def get_logprob_value(lp):
            return lp.logprob if hasattr(lp, "logprob") else lp

        exp_values = [np.exp(get_logprob_value(lp))
                      for lp in token_logprobs.values()]

        total = sum(exp_values)
        prob_dict = {k: float(v) / total for k,
                     v in zip(token_logprobs.keys(), exp_values)}

        word_probabilities.append(prob_dict)

    return word_probabilities


def load_dataset_my_way(dataset_path, output_path):
    """
    Load the dataset from the given path, handling both local and remote datasets.

    Assumption: the ordering of the rows in the saved dataset is the same as in the original dataset.
    """
    dataset = load_dataset(dataset_path)
    dataset = dataset["train_prefs"]

    already_processed_dataset = Dataset.from_dict(
        {k: [] for k in dataset.features})
    if os.path.exists(output_path):
        print(
            f"Output path {output_path} already exists. Filtering out already processed rows.")
        try:
            already_processed_dataset = load_from_disk(output_path)
        except Exception as e:
            print(
                f"Failed to load already processed dataset from {output_path}: {e}")
            already_processed_dataset = Dataset.from_dict(
                {k: [] for k in dataset.features})

        original_dataset_size = len(dataset)
        dataset = dataset.select(
            range(len(already_processed_dataset), original_dataset_size))

        print(f"{original_dataset_size - len(dataset)} rows were already processed. proceeding with {len(dataset)} rows.")

    return dataset, already_processed_dataset


if __name__ == "__main__":
    args = parse_args()

    print("=== Arguments ===")
    print(args)
    # exit()
    logger.info("Logging into HuggingFace")
    # setup(login_to_hf=True)

    if args.seed:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    logger.info(f"Loading {args.dataset_path}")

    dataset, already_processed_dataset = load_dataset_my_way(
        args.dataset_path, args.output_path)

    output_dataset = already_processed_dataset.to_list()

    if args.debug:
        logger.info(
            "Debug mode: only annotating completions for the first few prompts")
        dataset = dataset.select(range(100))
    logger.info(f"{len(dataset)}")

    logger.info(f"Using {args.model_name} for annotation")
    model = LLM(model=args.model_name, tensor_parallel_size=4)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    sampling_params = SamplingParams(
        max_tokens=64,  # args.max_tokens,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        logprobs=20,
    )

    all_raw_annotations = []
    logger.info("Annotating completions")

    batch_size = args.batch_size_to_annotate
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    os.makedirs(args.output_path, exist_ok=True)
    args_path = path.join(args.output_path, "args.json")
    with open(args_path, "w") as f_out:
        json.dump(vars(args), f_out)

    n_aspects = len(ASPECT2ANNOTATION_PROMPT.keys())
    aspects = list(ASPECT2ANNOTATION_PROMPT.keys())

    for batch_idx in tqdm(range(num_batches)):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(dataset))
        batch_dataset = dataset.select(range(start, end))

        # Build all_messages and all_metadata for this batch
        batch_messages = []
        batch_metadata = []
        for prompt_completion in batch_dataset:
            for aspect, annotation_prompt in ASPECT2ANNOTATION_PROMPT.items():
                messages = [
                    {
                        "role": "system",
                        "content": PREFERENCE_ANNOTATION_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": annotation_prompt.format(
                            prompt=prompt_completion["prompt"],
                            completion=prompt_completion[args.model_to_annotate][1]["content"]
                        ),
                    }
                ]
                batch_messages.append(messages)
                batch_metadata.append({
                    "prompt_id": prompt_completion["prompt_id"],
                    "model": args.model_to_annotate,
                    "aspect": aspect,
                })

        all_raw_outputs = get_response_texts(
            model, tokenizer, batch_messages, sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})

        all_raw_objects = [output["raw"] for output in all_raw_outputs]

        all_probabilities = calculate_probabilities(
            all_raw_objects, tokenizer, target_words=["1", "2", "3", "4", "5"])

        batch_outputs = []
        for i in range(len(batch_dataset)):
            new_row = {
                "prompt_id": batch_metadata[i * n_aspects]["prompt_id"],
                "model": batch_metadata[i * n_aspects]["model"],
                "annotation": {}
            }

            for j in range(n_aspects - 1):
                if batch_metadata[i * n_aspects + j]["prompt_id"] != batch_metadata[i * n_aspects + j + 1]["prompt_id"]:
                    raise ValueError(
                        "Aspects are not in the same order for all completions in the batch")
            if i != len(batch_dataset) - 1:
                if batch_metadata[i * n_aspects]["prompt_id"] == batch_metadata[(i + 1) * n_aspects]["prompt_id"]:
                    raise ValueError(
                        "Aspects are not in the same order for all completions in the batch 2")

            for j in range(n_aspects):
                new_row["annotation"][aspects[j]
                                      ] = all_probabilities[i * n_aspects + j]
            batch_outputs.append(new_row)

        output_dataset.extend(batch_outputs)
        Dataset.from_list(output_dataset).save_to_disk(args.output_path)
