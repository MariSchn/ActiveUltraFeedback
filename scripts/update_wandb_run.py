import wandb
import json
import os
from typing import Dict


def read_rm_scores(rm_output_dir: str) -> Dict[str, float]:
    results_path = os.path.join(rm_output_dir, "results.json")

    try:
        with open(results_path, "r") as f:
            rm_scores = json.load(f)
    except Exception:
        rm_scores = {}

    return rm_scores


def _read_dpo_score(results_path: str) -> float:
    try:
        with open(results_path, "r") as f:
            score = json.load(f)["metrics"][0]["primary_score"]
    except Exception:
        score = None

    return score


def read_dpo_scores(dpo_output_dir: str) -> Dict[str, float]:
    # fmt: off

    # {display_name: path}
    benchmarks = {
        "GSM8K": os.path.join(dpo_output_dir, "results", "gsm8k_tulu", "metrics.json"),
        "IF Eval": os.path.join(dpo_output_dir, "results", "ifeval_tulu", "metrics.json"),
        "Minerva Math": os.path.join(dpo_output_dir, "results", "minerva_math_tulu", "metrics.json"),
        "Truthful QA": os.path.join(dpo_output_dir, "results", "truthfulqa_tulu", "metrics.json"),
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
    # Read paths from environment variables. These should be set by the ./scripts/loop_train_eval.sbatch script
    run_id = os.getenv("LOOP_WANDB_RUN_ID")
    rm_output_dir = os.getenv("RM_OUTPUT_DIR")
    dpo_output_dir = os.getenv("DPO_OUTPUT_DIR")

    print("Found the following directories from environment variables:")
    print(f"LOOP_WANDB_RUN_ID={os.getenv('LOOP_WANDB_RUN_ID')}")
    print(f"RM_OUTPUT_DIR={os.getenv('RM_OUTPUT_DIR')}")
    print(f"DPO_OUTPUT_DIR={os.getenv('DPO_OUTPUT_DIR')}")

    # Get wandb run and its existing metrics
    run = wandb.init(id=run_id, project="loop", entity="ActiveUF", resume="must")
    existing_score_names = set(run.summary.keys()) if run.summary else set()

    # Read RM Scores
    rm_scores = read_rm_scores(rm_output_dir)
    if not rm_scores:
        print(
            f"\033[91mWARNING\033[0m: No Rewardbench scores found in RM dir: {os.path.join(rm_output_dir, 'results.json')}"
        )

    # Read DPO scores
    dpo_scores = read_dpo_scores(dpo_output_dir)
    if not dpo_scores:
        print(
            f"\033[91mWARNING\033[0m: No DPO scores found in DPO dir: {dpo_output_dir}"
        )

    # Add section prefixes
    log_dict = {}
    for key, value in rm_scores.items():
        log_dict[f"Rewardbench/{key}"] = value
    for key, value in dpo_scores.items():
        log_dict[f"DPO/{key}"] = value
    print(f"Candidate metrics to log to run {run_id}: {log_dict}")

    # Filter out scores that already exist. This can happen when re-running benchmarks because of a node failure.
    filtered_log_dict = {
        k: v for k, v in log_dict.items() if k not in existing_score_names
    }

    if filtered_log_dict:
        print(f"Logging new metrics to run {run_id}: {filtered_log_dict}")
        run.log(filtered_log_dict)
    else:
        print(f"All metrics already exist in run {run_id}. Skipping logging.")

    run.finish()
    print(f"Run {run_id} finished.")
