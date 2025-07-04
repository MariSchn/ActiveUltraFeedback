import argparse
import os.path as path

from datasets import Dataset, load_from_disk

from activeuf.configs import *
from activeuf.prompts import *
from activeuf.schemas import *
from activeuf.utils import *

"""
python -m activeuf.construct_inputs_for_bulk_annotation --dataset_path datasets/merged_completions --debug
"""

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="The path to the dataset with completions to be annotated")
    parser.add_argument("--output_path", type=str, help="Where to export the messages to be submitted to the annotator")
    parser.add_argument("--debug", action="store_true", help="If set, will only annotate the first 2 samples")
    args = parser.parse_args()

    if not args.output_path:
        args.output_path = f"datasets/inputs_for_bulk_annotation"
    assert not path.exists(args.output_path), f"Output path {args.output_path} already exists"

    return args

def construct_inputs_for_annotation(sample: dict) -> dict:

    inputs = []
    for completion in sample["completions"]:
        for aspect, aspect_prompt in ASPECT2ANNOTATION_PROMPT.items():
            inputs.append({
                "prompt_id": sample["prompt_id"],
                "model": completion["model"],
                "aspect": aspect,
                "messages": [
                    Message(
                        role="system", 
                        content=PREFERENCE_ANNOTATION_SYSTEM_PROMPT,
                    ).model_dump(),
                    Message(
                        role="user", 
                        content=f"{aspect_prompt}\n\nInstruction: {sample['prompt']}\n\nText: {completion['response_text']}",
                    ).model_dump(),
            ]})
        inputs.append({
            "prompt_id": sample["prompt_id"],
            "model": completion["model"],
            "aspect": "critique",
            "messages": [
                Message(
                    role="system", 
                    content=CRITIQUE_ANNOTATION_SYSTEM_PROMPT + "\n" + FEEDBACK_ANNOTATION_SYSTEM_PROMPT
                ).model_dump(),
                Message(
                    role="user", 
                    content=f"Instruction: {sample['prompt']}\n\nText: {completion['response_text']}",
                ).model_dump(),
        ]})
        
    return {"inputs": inputs}

if __name__ == "__main__":
    args = parse_args()
    
    dataset = load_from_disk(args.dataset_path)
    if args.debug:
        dataset = dataset.select(range(3))

    temp = dataset.map(construct_inputs_for_annotation)
    all_inputs = Dataset.from_list([_ for x in temp for _ in x["inputs"]])

    all_inputs.save_to_disk(args.output_path)