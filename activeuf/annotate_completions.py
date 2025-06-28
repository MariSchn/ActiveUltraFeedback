import argparse
import json
import re
import os.path as path

import vllm
from datasets import Dataset, load_from_disk
from transformers import Pipeline, AutoTokenizer, PreTrainedModel

from activeuf.configs import *
from activeuf.prompts import *
from activeuf.schemas import *
from activeuf.utils import *

logger = get_logger(__name__)

"""
This script is used to annotate the completions generated from the generate_completions.py script.
It uses a LLM as a judge to rate the completions based on the aspects defined in the configs.py file and provides critique/feedback for each completion.

Example run command:
    python -m activeuf.annotate_completions \
        --dataset_path datasets/allenai/ultrafeedback_binarized_cleaned/train_prefs-with-completions-sanitized \
        --model_name "meta-llama/Llama-3.2-1B-Instruct" \
        --part "first"
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="The path to the dataset with completions to be annotated")
    parser.add_argument("--model_name", type=str, required=True, help="The Huggingface path or API of the model to use for annotations (e.g. HuggingFaceTB/SmolLM2-135M-Instruct, gpt-4)")

    parser.add_argument("--seed", type=int, default=SEED, help="Seed for random sampling")
    parser.add_argument("--max_num_gpus", type=int, default=MAX_NUM_GPUS, help="The maximum number of GPUs to use")
    parser.add_argument("--model_class", type=str, default=DEFAULT_MODEL_CLASS, help="The class which is used to perform inference (e.g. transformers, pipeline, vllm)")
    
    parser.add_argument("--max_tokens", type=int, default=ANNOTATION_MAX_TOKENS, help="The maximum number of tokens for LLM responses")
    parser.add_argument("--temperature", type=float, default=ANNOTATION_TEMPERATURE, help="The temperature for sampling")
    parser.add_argument("--top_p", type=float, default=ANNOTATION_TOP_P, help="The top_p for sampling")

    parser.add_argument("--output_path", type=str, help="Where to export the annotated dataset")
    parser.add_argument("--debug", action="store_true", help="If set, will only annotate the first few samples")

    parser.add_argument("--download_dir", type=str, default="./hf_cache", help="Local path to model weights")
    parser.add_argument("--part", type=str, choices=["first", "second", "third", "fourth"], help="Which part of the dataset to annotate")
    args = parser.parse_args()

    if not args.output_path:
        args.output_path = f"{args.dataset_path.rstrip('/')}-annotated-{args.part}"
    assert not path.exists(args.output_path), f"Output path {args.output_path} already exists"

    return args

def annotate(
        dataset: Dataset, 
        model: str | vllm.LLM | Pipeline | PreTrainedModel, 
        tokenizer: AutoTokenizer | None,
        sampling_params: vllm.SamplingParams | None, 
    ) -> Dataset:
    """
    Annotates a given dataset with completions under the aspects defined in configs.py.
    The function uses a LLM as a judge to rate the completions on the aspects using prompts defined in prompts.py.
    It operates in place and modifies the dataset directly.

    Args:
        dataset (Dataset): The dataset to be annotated
        model (str | vllm.LLM | Pipeline | PreTrainedModel): The loaded model to be used for annotation (if using an API call, this should be None)
        tokenizer (AutoTokenizer | None): The tokenizer used for the model (if using an API call, this should be None)
        sampling_params (vllm.SamplingParams | None): The sampling parameters for the model. If None is provided the default parameters are used
    """
    # prepare new column for the annotated completions
    all_annotated_completions = dataset["completions"]

    # CRITIQUE ANNOTATION
    logger.info("Critiquing completions")
    for sample, annotated_completions in tqdm(zip(dataset, all_annotated_completions)):
        # identify completions that need an "overall" critique
        idxs_needing_annotation = [
            i for i, completion in enumerate(sample["completions"])
            if not completion["overall_score"]
        ]

        # construct messages for critique of each completion
        all_messages = []
        for i in idxs_needing_annotation:
            completion = sample["completions"][i]
            all_messages.append([
                Message(
                    role="system", 
                    content=SCORE_ANNOTATION_SYSTEM_PROMPT,
                ).model_dump(),
                Message(
                    role="user", 
                    content=f"Instruction: {sample['prompt']}\n\nText: {completion['response_text']}",
                ).model_dump(),
            ])

        # generate responses for all messages
        response_texts = get_response_texts(model, tokenizer, all_messages, sampling_params)
        
        # extract critiques from response texts (warn, but don't fail if parsing error)
        for i, response_text in zip(idxs_needing_annotation, response_texts):
            try:
                annotated_completions[i]["overall_score"] = response_text
            except:
                logger.info(f"Failed to critique a completion for prompt_id={sample['prompt_id']} on overall")
                logger.info(response_text)

    # replace existing completions with annotated completions
    dataset = dataset.remove_columns("completions")
    dataset = dataset.add_column("completions", all_annotated_completions)
    return dataset

if __name__ == "__main__":
    args = parse_args()

    logger.info("Logging into HuggingFace")
    setup(login_to_hf=True)

    if args.seed:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    logger.info(f"Loading {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.map(
        lambda x: PromptWithCompletions(**x).model_dump()
    )
    if args.debug:
        logger.info("Debug mode: only generating completions for the first 2 samples")
        dataset = dataset.select(range(100))

    n = len(dataset)
    m = (n // 4) + 1
    idxs = list(range(0, n+m, m))
    if args.part == "first":
        dataset = dataset.select(range(idxs[0], idxs[1]))
    elif args.part == "second":
        dataset = dataset.select(range(idxs[1], idxs[2]))
    elif args.part == "third":
        dataset = dataset.select(range(idxs[2], idxs[3]))
    elif args.part == "fourth":
        dataset = dataset.select(range(idxs[3], idxs[4]))
    logger.info(f"{n}, {len(dataset)}, {args.part}")

    logger.info(f"Using {args.model_name} for annotation")
    model, tokenizer = load_model(args.model_name, args.model_class, download_dir=args.download_dir)
    sampling_params = vllm.SamplingParams(
        max_tokens = args.max_tokens,
        temperature = args.temperature,
        top_p = args.top_p,      
    )

    logger.info("Annotating dataset")
    dataset = annotate(dataset, model, tokenizer, sampling_params)

    logger.info(f"Saving annotated dataset to {args.output_path}")
    dataset.save_to_disk(args.output_path)

    args_path = path.join(args.output_path, "args.json")
    with open(args_path, "w") as f_out:
        json.dump(vars(args), f_out)
