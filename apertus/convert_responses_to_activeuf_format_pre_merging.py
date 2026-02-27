import argparse
from datasets import load_from_disk, Dataset
import os.path as path

"""
Example usage:
python apertus/convert_responses_to_activeuf_format_pre_merging.py \
    -i /iopsstor/scratch/cscs/smarian/datasets/apertus/Dolci-Instruct-DPO/EuroLLM-1.7B-Instruct \
    -o ../../datasets/apertus/Dolci-Instruct-DPO/2a_converted_completions/EuroLLM-1.7B-Instruct \
    --cache_dir /tmp/hf-map-cache.arrow
"""

def convert_responses_to_activeuf_format(x: dict, model_name: str) -> dict:
    y = {}

    # setting this to dummy value instead of throwing it away, to be safe
    y["source"] = ""

    y["prompt_id"] = x["prompt_id"]
    y["prompt"] = x["chosen"][0]["content"]
    y["completions"] = [
        {
            # setting these to dummy values instead of throwing them away, to be safe
            "annotations": [],
            "critique": "",
            "overall_score": "",
            "system_prompt": "",
            "principle": "", 
            "messages": [],

            "model": model_name,
            "response_text": x["response"] if x["response"] else "",  # str, not None
        }
    ]
    
    return y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        "-i",
        type=str,
        required=True,
        help="Path to the input dataset (load_from_disk-compatible folder).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        required=True,
        help="Path to save the converted dataset (save_to_disk folder).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to use as cache during mapping (cache_file_name folder).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = path.basename(args.input_path)

    ds: Dataset = load_from_disk(args.input_path)
    original_column_names = ds.column_names

    converted: Dataset = ds.map(
        convert_responses_to_activeuf_format,
        fn_kwargs={"model_name": model_name},

        remove_columns=original_column_names,
        cache_file_name=args.cache_dir,
        load_from_cache_file=False,
    )
    converted.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()