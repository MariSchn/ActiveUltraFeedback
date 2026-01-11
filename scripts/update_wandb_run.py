import wandb
import json
import os
import argparse
import csv
from typing import Dict, Optional


def read_rm_scores(rm_output_dir: str) -> Dict[str, float]:
    results_path = os.path.join(rm_output_dir, "metrics.json")

    try:
        with open(results_path, "r") as f:
            rm_scores = json.load(f)
    except Exception:
        rm_scores = {}

    return rm_scores


def _read_single_score(results_path: str) -> Optional[float]:
    """Helper function to read a score from a specific file path (json or csv)."""
    try:
        if results_path.endswith(".json"):
            with open(results_path, "r") as f:
                score = json.load(f)["metrics"][0]["primary_score"]

        elif results_path.endswith(".csv"):
            with open(results_path, "r") as f:
                reader = csv.DictReader(f)
                first_row = next(reader)
                score = float(first_row["length_controlled_winrate"]) / 100.0
        else:
            score = None
    except Exception:
        score = None

    return score


def read_benchmark_scores(output_dir: str, model_label: str) -> Dict[str, float]:
    """
    Reads benchmark scores from a given output directory.
    model_label: Used for printing warnings (e.g., 'DPO', 'IPO', 'SIMPO')
    """
    
    # {display_name: path}
    benchmarks = {
        "GSM8K": os.path.join(output_dir, "results", "gsm8k_tulu", "metrics.json"),
        "IF Eval": os.path.join(output_dir, "results", "ifeval_tulu", "metrics.json"),
        # "Minerva Math": os.path.join(output_dir, "results", "minerva_math_tulu", "metrics.json"),
        "Truthful QA": os.path.join(output_dir, "results", "truthfulqa_tulu", "metrics.json"),
        "Alpaca Eval": os.path.join(output_dir, "results", "alpaca_eval", "activeuf", "leaderboard.csv"),
    }

    # Read scores
    scores = {}
    for display_name, path in benchmarks.items():
        score = _read_single_score(path)

        if score is not None:
            scores[display_name] = score
        else:
            print(
                f"\033[91mWARNING\033[0m: No {display_name} scores found in {model_label} dir: {path}"
            )

    # Calculate mean
    if scores and len(scores) == len(benchmarks):
        mean = 0.0
        for _, score in scores.items():
            mean += score
        mean /= len(scores)
        scores["Mean"] = mean
    else:
        # Avoid division by zero or partial means if that's undesired
        if scores:
            print(f"\033[93mWARNING\033[0m: {model_label}/Mean calculated on partial results ({len(scores)}/{len(benchmarks)} found).")
            # Optional: Calculate partial mean if you want, currently logic skips it to be safe
            # mean = sum(scores.values()) / len(scores)
            # scores["Mean"] = mean
        else:
             print(f"\033[91mWARNING\033[0m: No scores found for {model_label}.")

    return scores


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Update WandB run with RM and DPO (optionally IPO and SIMPO) Evaluation metrics"
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="WandB run ID to update"
    )
    # CHANGED: required=False so script doesn't crash if RM/DPO is missing
    parser.add_argument(
        "--rm_output_dir", type=str, required=False, help="Reward model output directory"
    )
    parser.add_argument(
        "--dpo_output_dir", type=str, required=False, help="DPO output directory"
    )
    # Optional Arguments
    parser.add_argument(
        "--ipo_output_dir",
        type=str,
        required=False,
        help="IPO output directory (Optional)",
    )
    parser.add_argument(
        "--simpo_output_dir",
        type=str,
        required=False,
        help="SimPO output directory (Optional)",
    )

    parser.add_argument(
        "--project", type=str, default="loop", help="WandB project name (default: loop)"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="ActiveUF",
        help="WandB entity name (default: ActiveUF)",
    )
    args = parser.parse_args()

    print(f"Updating Run: {args.run_id}")

    # Get wandb run and its existing metrics
    run = wandb.init(
        id=args.run_id, project=args.project, entity=args.entity, resume="must"
    )
    
    log_dict = {}

    # 1. Read RM Scores (If provided)
    if args.rm_output_dir:
        rm_scores = read_rm_scores(args.rm_output_dir)
        if not rm_scores:
            print(
                f"\033[91mWARNING\033[0m: RM directory provided but no scores found: {args.rm_output_dir}"
            )
        for key, value in rm_scores.items():
            log_dict[f"Rewardbench/{key}"] = value
    else:
        print("No RM output directory provided. Skipping RM scores.")

    # 2. Read DPO scores (If provided)
    if args.dpo_output_dir:
        dpo_scores = read_benchmark_scores(args.dpo_output_dir, "DPO")
        if not dpo_scores:
            print(
                f"\033[91mWARNING\033[0m: DPO directory provided but no scores found: {args.dpo_output_dir}"
            )
        for key, value in dpo_scores.items():
            log_dict[f"DPO/{key}"] = value
            print(f"DPO Score - {key}: {value}")
    else:
        print("No DPO output directory provided. Skipping DPO scores.")

    # 3. Read IPO scores (If provided)
    if args.ipo_output_dir:
        ipo_scores = read_benchmark_scores(args.ipo_output_dir, "IPO")
        for key, value in ipo_scores.items():
            log_dict[f"IPO/{key}"] = value
            print(f"IPO Score - {key}: {value}")

    # 4. Read SimPO scores (If provided)
    if args.simpo_output_dir:
        simpo_scores = read_benchmark_scores(args.simpo_output_dir, "SIMPO")
        for key, value in simpo_scores.items():
            log_dict[f"SIMPO/{key}"] = value
            print(f"SIMPO Score - {key}: {value}")

    # Calculate Main Final Score (DPO + RM) / 2
    # Only calculate if BOTH exist in the current log_dict
    if "DPO/Mean" in log_dict and "Rewardbench/Mean" in log_dict:
        log_dict["Final Score/Mean"] = (
            log_dict["DPO/Mean"] + log_dict["Rewardbench/Mean"]
        ) / 2

    # Calculate Final Score including IPO and SIMPO if available
    # This averages whatever means are present
    sum_scores = 0.0
    count_scores = 0
    for prefix in ["DPO", "Rewardbench", "IPO", "SIMPO"]:
        key = f"{prefix}/Mean"
        if key in log_dict:
            sum_scores += log_dict[key]
            count_scores += 1
    
    if count_scores > 0:
        log_dict["Final Score/Mean (All)"] = sum_scores / count_scores

    print(f"Candidate metrics to log to run {args.run_id}: {log_dict}")

    if log_dict:
        run.log(log_dict)
        print(f"Successfully logged metrics to {args.run_id}")
    else:
        print(f"No metrics found to log for {args.run_id}.")

    run.finish()