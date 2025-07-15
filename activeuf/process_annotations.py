import argparse
from collections import defaultdict
import glob
import os.path as path
import regex as re

from datasets import concatenate_datasets, load_from_disk, Dataset

from activeuf.configs import *
from activeuf.prompts import *
from activeuf.schemas import *
from activeuf.utils import *

"""
python -m activeuf.process_annotations --bulk_annotations_path datasets/bulk_annotations --completions_path datasets/merged_completions --debug
"""

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions_path", type=str, required=True, help="The path to the completions dataset")
    parser.add_argument("--bulk_annotations_path", type=str, required=True, help="The path to bulk annotations dataset")
    parser.add_argument("--output_path", type=str, help="Where to export the processed annotations")

    parser.add_argument("--debug", action="store_true", help="If set, will limit many operations for speed")
    args = parser.parse_args()

    if not args.output_path:
        args.output_path = f"datasets/annotations"
    assert not path.exists(args.output_path), f"Output path {args.output_path} already exists"

    return args

def process_aspect_annotation(x: dict) -> dict:
    matches = re.search(
        ASPECT2ANNOTATION_PATTERN[x["aspect"]], 
        x["raw_annotations"], 
        re.DOTALL | re.I,
    )
    if matches is not None:
        groups = matches.groups()
        if len(groups) == 2:
            return {
                "type": "",
                "type_rationale": "",
                "rating": groups[0].strip(),
                "rating_rationale": groups[1].strip(),
            }
        elif len(groups) == 4:
            return {
                "type": groups[0].strip(),
                "type_rationale": groups[1].strip(),
                "rating": groups[2].strip(),
                "rating_rationale": groups[3].strip(),
            }
    
    return {
        "type": "",
        "type_rationale": "",
        "rating": "",
        "rating_rationale": "",
    }

def process_critique_annotation(x: dict) -> dict:
    matches = re.search(
        FEEDBACK_ANNOTATION_PATTERN, x["raw_annotations"], re.DOTALL | re.I
    )
    groups = matches.groups()
    return {
        "critique": groups[0].strip(), 
        "overall_score": groups[1].strip(),
    }
        
if __name__ == "__main__":
    args = parse_args()

    # load completions
    completions_ds = load_from_disk(args.completions_path)
    if args.debug:
        completions_ds = completions_ds.select(range(10_000))

    # load bulk annotations (in parts, if input_path is not itself a dataset but instead contains multiple datasets)
    try:
        temp = load_from_disk(args.bulk_annotations_path)
    except FileNotFoundError:
        parts = []
        for filepath in sorted(glob.glob(f"{args.bulk_annotations_path}/*")):
            parts.append(load_from_disk(filepath))
        temp = concatenate_datasets(parts)
    
    # keep only the relevant annotations, then split into critique and aspect annotations
    prompt_ids = set(completions_ds["prompt_id"])
    temp = temp.filter(lambda x: x["prompt_id"] in prompt_ids)
    critique_annotations_ds = temp.filter(lambda x: x["aspect"] == "critique")
    aspect_annotations_ds = temp.filter(lambda x: x["aspect"] != "critique")

    # create mapper from (prompt_id, model) to prompt
    prompt_id2annotated_prompt = {
        x["prompt_id"]: x for x in completions_ds
    }

    # update prompts' completions with critique annotations
    for x in critique_annotations_ds:
        try:
            prompt = prompt_id2annotated_prompt[x['prompt_id']]
            annotation = process_critique_annotation(x)
            for i, completion in enumerate(prompt["completions"]):
                if completion["model"] == x["model"]:
                    print(x["model"])

                    temp = completion.copy()
                    temp.update(annotation)
                    prompt["completions"][i] = temp
                    break
        except:
            logger.info(f"Failed to critique a completion for {x['prompt_id']=}")

    # process aspect annotations
    aspect_annotations_ds = aspect_annotations_ds.map(process_aspect_annotation)

    # structure as (prompt_id, model) -> list of annotations dict
    pair_id2aspect_annotations = defaultdict(list)
    for x in tqdm(aspect_annotations_ds):
        pair_id2aspect_annotations[(x["prompt_id"], x["model"])].append({
            "aspect": x["aspect"],
            "text": x["raw_annotations"],
            "type": x["type"],
            "type_rationale": x["type_rationale"],
            "rating": x["rating"],
            "rating_rationale": x["rating_rationale"],
        })
    
    # update prompts' completions with aspect annotations
    for x in tqdm(completions_ds):
        annotated_completions = []
        for completion in x["completions"]:
            pair_id = (x["prompt_id"], completion["model"])
            completion["annotations"] = pair_id2aspect_annotations.get(pair_id, [])
            annotated_completions.append(completion)

        
        prompt_id2annotated_prompt[x["prompt_id"]]["completions"] = annotated_completions

    # convert annotated prompts to dataset and export
    out_ds = Dataset.from_list(list(prompt_id2annotated_prompt.values()))
    out_ds.save_to_disk(args.output_path)