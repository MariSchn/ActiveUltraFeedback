

from argparse import ArgumentParser
from datasets import load_dataset, load_from_disk, DatasetDict
import numpy as np
import os


def main(ars):
    # If input_file is a directory, combine all sub-datasets
    if ars.input_file_is_dir:
        all_chosen = []
        all_rejected = []
        # List all subdirectories in the input directory
        for entry in sorted(os.listdir(ars.input_file)):
            partition_dir = os.path.join(ars.input_file, entry)
            print(f"Processing partition directory: {partition_dir}")
            if os.path.isdir(partition_dir):
                # Look for exactly one subdirectory inside partition_dir
                subdirs = [d for d in os.listdir(partition_dir) if os.path.isdir(os.path.join(partition_dir, d))]
                if len(subdirs) != 1:
                    print(f"Warning: {partition_dir} does not contain exactly one subdirectory, skipping.")
                    continue
                dataset_dir = os.path.join(partition_dir, subdirs[0])
                try:
                    ds = load_from_disk(dataset_dir)
                    all_chosen.extend(ds["chosen_annotations"])
                    all_rejected.extend(ds["rejected_annotations"])
                    print(f"Loaded {len(ds['chosen_annotations'])} chosen and {len(ds['rejected_annotations'])} rejected annotations from {dataset_dir}")
                except Exception as e:
                    print(f"Warning: Could not load {dataset_dir}: {e}")
                
        dataset = {"chosen_annotations": all_chosen, "rejected_annotations": all_rejected}
        print(f"Loaded {len(all_chosen)} chosen and {len(all_rejected)} rejected annotations from directory {ars.input_file}")
    else:
        dataset = load_from_disk(ars.input_file)
        print(dataset)
        print(dataset["chosen_annotations"][5])
        print(dataset["rejected_annotations"][5])

    # Choose score calculator
    if ars.score_mode == "probs":
        score_fn = score_from_probs
    else:
        score_fn = score_from_direct

    chosen_scores = []
    rejected_scores = []
    chosen_parse_errors = 0
    rejected_parse_errors = 0
    win_count = 0  # absolute wins
    tie_count = 0  # absolute ties
    total_count = 0

    # For aspect value tracking
    aspect_value_sets_chosen = {}
    aspect_value_sets_rejected = {}

    # Get all aspects from the first annotation (if any)
    aspects = []
    if dataset["chosen_annotations"]:
        aspects = list(dataset["chosen_annotations"][0].keys())

    for aspect in aspects:
        aspect_value_sets_chosen[aspect] = set()
        aspect_value_sets_rejected[aspect] = set()

    # Counters for annotations that are fully one-hot across all aspects (probs mode)
    chosen_all_onehot = 0
    rejected_all_onehot = 0

    # Build a Hugging Face Dataset for efficient mapping
    from datasets import Dataset as HfDataset
    if isinstance(dataset, dict):
        hf_ds = HfDataset.from_dict({"chosen": dataset["chosen_annotations"], "rejected": dataset["rejected_annotations"]})
    else:
        # dataset might be an HF Dataset or DatasetDict; try to extract columns
        try:
            chosen_col = dataset["chosen_annotations"]
            rejected_col = dataset["rejected_annotations"]
            hf_ds = HfDataset.from_dict({"chosen": chosen_col, "rejected": rejected_col})
        except Exception:
            # Fallback: assume dataset is already suitable
            hf_ds = dataset

    # Processing function for map
    def process_example(example):
        chosen_ann = example.get("chosen")
        rejected_ann = example.get("rejected")
        c_score, c_errs = score_fn(chosen_ann)
        r_score, r_errs = score_fn(rejected_ann)

        # argmax/cleaned values per aspect for aggregation
        argmax_c = {}
        argmax_r = {}
        if ars.score_mode == "probs":
            for aspect, val in (chosen_ann or {}).items():
                if isinstance(val, dict):
                    try:
                        arg = max(val.items(), key=lambda x: x[1])[0]
                        argmax_c[aspect] = str(arg)
                    except Exception:
                        pass
            for aspect, val in (rejected_ann or {}).items():
                if isinstance(val, dict):
                    try:
                        arg = max(val.items(), key=lambda x: x[1])[0]
                        argmax_r[aspect] = str(arg)
                    except Exception:
                        pass
        else:
            for aspect, val in (chosen_ann or {}).items():
                if isinstance(val, str) and "</think>\n\n" in val:
                    val = val.split("</think>\n\n", 1)[1].strip()
                argmax_c[aspect] = str(val)
            for aspect, val in (rejected_ann or {}).items():
                if isinstance(val, str) and "</think>\n\n" in val:
                    val = val.split("</think>\n\n", 1)[1].strip()
                argmax_r[aspect] = str(val)

        # one-hot checks for the whole annotation (all aspects)
        chosen_onehot = False
        rejected_onehot = False
        if ars.score_mode == "probs":
            def ann_all_onehot(ann):
                if not isinstance(ann, dict):
                    return False
                for aspect, v in ann.items():
                    if not isinstance(v, dict) or len(v) == 0:
                        return False
                    vals = list(v.values())
                    ones = sum(1 for x in vals if np.isclose(x, 1.0))
                    zeros = sum(1 for x in vals if np.isclose(x, 0.0))
                    if ones != 1 or ones + zeros != len(vals):
                        return False
                return True

            chosen_onehot = ann_all_onehot(chosen_ann)
            rejected_onehot = ann_all_onehot(rejected_ann)

        return {
            "chosen_score": c_score,
            "rejected_score": r_score,
            "chosen_parse_errors": c_errs,
            "rejected_parse_errors": r_errs,
            "argmax_chosen": argmax_c,
            "argmax_rejected": argmax_r,
            "chosen_all_onehot": chosen_onehot,
            "rejected_all_onehot": rejected_onehot,
        }

    num_proc = min(8, (os.cpu_count() or 1))
    try:
        mapped = hf_ds.map(process_example, remove_columns=None, num_proc=num_proc)
    except Exception:
        mapped = hf_ds.map(process_example, remove_columns=None)

    # Aggregate results from mapped dataset
    chosen_scores = mapped["chosen_score"]
    rejected_scores = mapped["rejected_score"]
    chosen_parse_errors = int(sum(mapped["chosen_parse_errors"]))
    rejected_parse_errors = int(sum(mapped["rejected_parse_errors"]))
    chosen_all_onehot = int(sum(1 for v in mapped["chosen_all_onehot"] if v))
    rejected_all_onehot = int(sum(1 for v in mapped["rejected_all_onehot"] if v))
    both_all_onehot = int(sum(1 for c, r in zip(mapped["chosen_all_onehot"], mapped["rejected_all_onehot"]) if c and r))

    # Compute wins/ties/valid comparisons
    win_count = 0
    tie_count = 0
    total_count = 0
    for c_score, r_score in zip(chosen_scores, rejected_scores):
        if c_score is not None and r_score is not None:
            total_count += 1
            if c_score > r_score:
                win_count += 1
            elif c_score == r_score:
                tie_count += 1

    # Build aspect value sets from argmax columns
    for arg_c, arg_r in zip(mapped["argmax_chosen"], mapped["argmax_rejected"]):
        if isinstance(arg_c, dict):
            for aspect, val in arg_c.items():
                aspect_value_sets_chosen.setdefault(aspect, set()).add(str(val))
        if isinstance(arg_r, dict):
            for aspect, val in arg_r.items():
                aspect_value_sets_rejected.setdefault(aspect, set()).add(str(val))

        # If in probs mode, check whether this entire annotation is one-hot for all aspects
        if ars.score_mode == "probs":
            def ann_is_all_onehot(ann):
                for aspect in aspects:
                    val = ann.get(aspect)
                    if not isinstance(val, dict) or len(val) == 0:
                        return False
                    vals = list(val.values())
                    # must have exactly one value ~1.0 and the rest ~0.0
                    ones = sum(1 for v in vals if np.isclose(v, 1.0))
                    zeros = sum(1 for v in vals if np.isclose(v, 0.0))
                    if ones != 1 or ones + zeros != len(vals):
                        return False
                return True

            if ann_is_all_onehot(chosen_ann):
                chosen_all_onehot += 1
            if ann_is_all_onehot(rejected_ann):
                rejected_all_onehot += 1

    print(f"Num samples: {len(chosen_scores)}")
    print(f"Total chosen parse errors: {chosen_parse_errors}")
    print(f"Total rejected parse errors: {rejected_parse_errors}")
    print(f"Num valid comparisons: {total_count}")
    print(f"Num absolute wins (chosen > rejected): {win_count}")
    print(f"Num ties (chosen == rejected): {tie_count}")
    win_score = win_count
    tie_score = tie_count
    if total_count > 0:
        win_rate = (win_score + tie_score / 2) / total_count
        print(f"Win rate (wins + 0.5 * ties): {win_rate:.4f}")
    else:
        print("No valid comparisons for win rate.")

    if ars.score_mode == "probs":
        print(f"Num chosen responses with all aspects one-hot: {chosen_all_onehot}")
        print(f"Num rejected responses with all aspects one-hot: {rejected_all_onehot}")



def score_from_probs(annotation_dict):
    # annotation_dict: {aspect: {1: p1, 2: p2, ...}}
    aspect_scores = []
    parse_errors = 0
    for aspect, prob_dict in annotation_dict.items():
        try:
            items = sorted(((int(k), v) for k, v in prob_dict.items()), key=lambda x: x[0])
            expectation = sum(k * v for k, v in items)
            aspect_scores.append(expectation)
        except Exception:
            aspect_scores.append(None)
            parse_errors += 1
    valid = [x for x in aspect_scores if x is not None]
    if valid:
        return float(np.mean(valid)), parse_errors
    return None, parse_errors

def score_from_direct(annotation_dict):
    # annotation_dict: {aspect: string/int}
    aspect_scores = []
    parse_errors = 0
    for aspect, val in annotation_dict.items():
        try:
            # If reasoning is present, extract answer after '</think>\n\n'
            if isinstance(val, str) and "</think>\n\n" in val:
                val = val.split("</think>\n\n", 1)[1].strip()
            aspect_scores.append(int(val))
        except Exception:
            aspect_scores.append(None)
            parse_errors += 1
    valid = [x for x in aspect_scores if x is not None]
    if valid:
        return float(np.mean(valid)), parse_errors
    return None, parse_errors

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file or directory containing game data.")
    parser.add_argument("--input_file_is_dir", action="store_true", help="Treat input_file as a directory of datasets to combine.")
    parser.add_argument("--score_mode", type=str, default="probs", choices=["probs", "direct"], help="How to calculate scores: 'probs' (default, expectation over probabilities), or 'direct' (parse int from string)")
    args = parser.parse_args()
    main(args)