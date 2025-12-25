import wandb
import json
import os
import argparse
import csv
from typing import Dict


def read_rm_scores(rm_output_dir: str) -> Dict[str, float]:
    results_path = os.path.join(rm_output_dir, "metrics.json")

    try:
        with open(results_path, "r") as f:
            rm_scores = json.load(f)
    except Exception:
        rm_scores = {}

    return rm_scores


def _read_dpo_score(results_path: str) -> float:
    try:
        if results_path.endswith(".json"):
            with open(results_path, "r") as f:
                score = json.load(f)["metrics"][0]["primary_score"]

        elif results_path.endswith(".csv"):
            with open(results_path, "r") as f:
                reader = csv.DictReader(f)
                first_row = next(reader)
                score = float(first_row["length_controlled_winrate"]) / 100.0
    except Exception:
        score = None

    return score


def read_dpo_scores(dpo_output_dir: str) -> Dict[str, float]:
    # fmt: off

    # {display_name: path}
    benchmarks = {
        "GSM8K": os.path.join(dpo_output_dir, "results", "gsm8k_tulu", "metrics.json"),
        "IF Eval": os.path.join(dpo_output_dir, "results", "ifeval_tulu", "metrics.json"),
        # "Minerva Math": os.path.join(dpo_output_dir, "results", "minerva_math_tulu", "metrics.json"),
        "Truthful QA": os.path.join(dpo_output_dir, "results", "truthfulqa_tulu", "metrics.json"),
        "Alpaca Eval": os.path.join(dpo_output_dir, "results", "alpaca_eval", "activeuf", "leaderboard.csv"),
    }

    # Read scores
    dpo_scores = {}
    for display_name, path in benchmarks.items():
        score = _read_dpo_score(path)

        if score:
            dpo_scores[display_name] = score
        else:
            print(
                f"\033[91mWARNING\033[0m: No {display_name} scores found in DPO dir: {path}"
            )

    # Calculate mean
    if dpo_scores and len(dpo_scores) == len(benchmarks):
        mean = 0.0
        for _, score in dpo_scores.items():
            mean += score
        mean /= len(dpo_scores)
        dpo_scores["Mean"] = mean
    else:
        print(f"\033[91mWARNING\033[0m: No DPO/Mean score will be calculated, as not all DPO scores were found. Found {list(dpo_scores.keys())} but expected {list(benchmarks.keys())}. Missing {list(set(benchmarks.keys()) - set(dpo_scores.keys()))}")

    return dpo_scores


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Update WandB run with RM and DPO evaluation metrics"
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="WandB run ID to update"
    )
    parser.add_argument(
        "--rm_output_dir", type=str, required=True, help="Reward model output directory"
    )
    parser.add_argument(
        "--dpo_output_dir", type=str, required=True, help="DPO output directory"
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

    print("Received the following arguments:")
    print(f"run_id={args.run_id}")
    print(f"rm_output_dir={args.rm_output_dir}")
    print(f"dpo_output_dir={args.dpo_output_dir}")
    print(f"project={args.project}")
    print(f"entity={args.entity}")

    # Get wandb run and its existing metrics
    run = wandb.init(
        id=args.run_id, project=args.project, entity=args.entity, resume="must"
    )
    existing_score_names = set(run.summary.keys()) if run.summary else set()

    # Read RM Scores
    rm_scores = read_rm_scores(args.rm_output_dir)
    if not rm_scores:
        print(
            f"\033[91mWARNING\033[0m: No Rewardbench scores found in RM dir: {os.path.join(args.rm_output_dir, 'results.json')}"
        )

    # Read DPO scores
    dpo_scores = read_dpo_scores(args.dpo_output_dir)
    if not dpo_scores:
        print(
            f"\033[91mWARNING\033[0m: No DPO scores found in DPO dir: {args.dpo_output_dir}"
        )

    # Add section prefixes
    log_dict = {}
    for key, value in rm_scores.items():
        log_dict[f"Rewardbench/{key}"] = value
    for key, value in dpo_scores.items():
        log_dict[f"DPO/{key}"] = value
        print(f"DPO Score - {key}: {value}")

    if "DPO/Mean" in log_dict and "Rewardbench/Mean" in log_dict:
        log_dict["Final Score/Mean"] = (
            log_dict["DPO/Mean"] + log_dict["Rewardbench/Mean"]
        ) / 2

    print(f"Candidate metrics to log to run {args.run_id}: {log_dict}")

    # Filter out scores that already exist. This can happen when re-running benchmarks because of a node failure.
    filtered_log_dict = {
        k: v
        for k, v in log_dict.items()  # if k not in existing_score_names
    }

    if filtered_log_dict:
        print(f"Logging new metrics to run {args.run_id}: {filtered_log_dict}")
        run.log(filtered_log_dict)
    else:
        print(f"All metrics already exist in run {args.run_id}. Skipping logging.")

    run.finish()
    print(f"Run {args.run_id} finished.")
