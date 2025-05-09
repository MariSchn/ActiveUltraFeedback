import argparse
import json
import re
import os.path as path

from datasets import Dataset, load_from_disk
from vllm import LLM, SamplingParams
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
        --dataset_path datasets/allenai/ultrafeedback_binarized_cleaned/test_prefs-with-completions-merged \
        --model_name "meta-llama/Llama-3.2-1B-Instruct"
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

    parser.add_argument("--output_path", type=str, help="Where to export the annotated dataset")
    parser.add_argument("--debug", action="store_true", help="If set, will only annotate the first 2 samples")
    args = parser.parse_args()

    if not args.output_path:
        args.output_path = f"{args.dataset_path.rstrip('/')}-annotated"
    assert not path.exists(args.output_path), f"Output path {args.output_path} already exists"

    return args

# def parse_annotation_from_response_text(response_text: str, aspect: str) -> dict[str, str]:
#     """
#     Processes the response from an evaluation/annotation model and extracts the ratings and rationales for each completion from the response.
#     As it is not always guaranteed that the response follows the expected format, the function will raise an error if the response does not follow the expected format.

#     Preference annotation refers to the rating in regards to one specific aspect, e.g. instruction following, truthfulness, etc. on a scale from 1 to 5.

#     Args:
#         response_text (str): The response from the evaluation/annotation model
#         aspect (str): The aspect with which the response text was annotated
#     Returns:
#         dict[str, str]: A dictionary that follows the Annotation schema
#     """
#     annotation_pattern = ASPECT2ANNOTATION_PATTERN[aspect]
#     matches = re.search(annotation_pattern, response_text, re.DOTALL | re.I)
#     groups = matches.groups()

#     return Annotation(
#         aspect=aspect,
#         annotator_name=args.model_name,

#         type=groups[-4] if len(groups) == 4 else None,
#         type_rationale=groups[-3] if len(groups) == 4 else None,
#         rating=groups[-2],
#         rating_rationale=groups[-1],
#     ).model_dump()

# def parse_critique_from_response_text(response_text: str) -> dict[str, str]:
#     matches = re.search(FEEDBACK_ANNOTATION_PATTERN, response_text, re.DOTALL | re.I)
#     groups = matches.groups()

#     return {"critique": groups[0], "overall_score": groups[1]}

def annotate(
        dataset: Dataset, 
        model: str | LLM | Pipeline | PreTrainedModel, 
        tokenizer: AutoTokenizer | None,
        sampling_params: SamplingParams | None, 
    ) -> Dataset:
    """
    Annotates a given dataset with completions under the aspects defined in configs.py.
    The function uses a LLM as a judge to rate the completions on the aspects using prompts defined in prompts.py.
    It operates in place and modifies the dataset directly.

    Args:
        dataset (Dataset): The dataset to be annotated
        model (str | LLM | Pipeline | PreTrainedModel): The loaded model to be used for annotation (if using an API call, this should be None)
        tokenizer (AutoTokenizer | None): The tokenizer used for the model (if using an API call, this should be None)
        sampling_params (SamplingParams | None): The sampling parameters for the model. If None is provided the default parameters are used
    """
    # prepare new column for the annotated completions
    all_annotated_completions = dataset["completions"]

    # # ASPECT ANNOTATION # disabled for now
    # logger.info("Annotating completions on aspects")
    # aspects = list(ASPECT2ANNOTATION_PROMPT)
    # for aspect in aspects:
    #     aspect_annotation_prompt = ASPECT2ANNOTATION_PROMPT[aspect]
    #     for sample, annotated_completions in zip(dataset, all_annotated_completions):
            
    #         # identify completions that need annotation for this aspect
    #         idxs_needing_annotation = [
    #             i for i, completion in enumerate(sample["completions"])
    #             if aspect not in {_["aspect"] for _ in completion["annotations"]}
    #         ]

    #         # construct messages for annotation of each completion
    #         all_messages = []
    #         for i in idxs_needing_annotation:
    #             completion = sample["completions"][i]
    #             all_messages.append([
    #                 Message(
    #                     role="system", 
    #                     content=PREFERENCE_ANNOTATION_SYSTEM_PROMPT,
    #                 ).model_dump(),
    #                 Message(
    #                     role="user", 
    #                     content=f"{aspect_annotation_prompt}\n\nInstruction: {sample['prompt']}\n\nText: {completion['response_text']}",
    #                 ).model_dump(),
    #             ])
            
    #         # generate responses for all messages
    #         response_texts = get_response_texts(model, tokenizer, all_messages, sampling_params)

    #         # extract annotations from response texts (warn, but don't fail if parsing error)
    #         for i, response_text in zip(idxs_needing_annotation, response_texts):
    #             try:
    #                 annotation = parse_annotation_from_response_text(response_text, aspect)
    #                 annotated_completions[i]["annotations"].append(annotation)
    #             except:
    #                 logger.info(f"Failed to annotate a completion for prompt_id={sample['prompt_id']} on aspect={aspect}")
    #                 logger.info(response_text)
        
    #     if args.debug:
    #         logger.info("Debug mode: only annotating on one aspect")
    #         break

    # CRITIQUE ANNOTATION
    logger.info("Critiquing completions")
    for sample, annotated_completions in zip(dataset, all_annotated_completions):
        # identify completions that need an "overall" critique
        idxs_needing_annotation = [
            i for i, completion in enumerate(sample["completions"])
            if not completion["critique"] or not completion["overall_score"]
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
                # critique = parse_critique_from_response_text(response_text)
                # annotated_completions[i].update(critique)
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
    dataset = load_from_disk(args.dataset_path).map(
        lambda x: PromptWithCompletions(**x).model_dump()
    )
    if args.debug:
        logger.info("Debug mode: only generating completions for the first 2 samples")
        dataset = dataset.select(range(2))

    logger.info(f"Using {args.model_name} for annotation")
    model, tokenizer = load_model(args.model_name, args.model_class)
    sampling_params = SamplingParams(
        max_tokens = 10,
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