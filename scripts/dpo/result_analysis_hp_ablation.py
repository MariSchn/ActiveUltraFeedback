import os
import json
import argparse
import pandas as pd
import re
import numpy as np

# --- CONFIGURATION: BASELINE SCORES (SFT) ---
SFT_SCORES = {
    "gsm8k": 0.7589,
    "ifeval": 0.7153,
    "truthfulqa": 0.4676,
    "minerva_math": 0.3092,
}


def parse_model_name(name):
    meta = {}

    # 1. Extract Job ID
    match_job = re.match(r"^(\d+)-", name)
    if match_job:
        meta["Job ID"] = int(match_job.group(1))
        rest = name[len(match_job.group(0)) :]
    else:
        meta["Job ID"] = -1
        rest = name

    # 2. Extract Acquisition Function
    parts = rest.split("-")
    acq_block = parts[0]
    meta["Method"] = acq_block.split("_")[0] if "_" in acq_block else acq_block

    # 3. Detect Training Mode (Full vs LoRA)
    if "full" in name.lower():
        meta["Type"] = "Full"
        meta["LoRA R"] = "-"
        meta["LoRA A"] = "-"
    else:
        meta["Type"] = "LoRA"
        meta["LoRA R"] = np.nan
        meta["LoRA A"] = np.nan

    # 4. Extract Hyperparameters
    hp_string = "-".join(parts[1:])
    patterns = {
        "LR": r"lr([0-9\.eE-]+)",
        "Lambda": r"sg([\d\.]+)",
        "Beta": r"b([\d\.]+)",
        "LoRA R": r"loraR(\d+)",
        "LoRA A": r"loraA(\d+)",
    }

    for hp_name, pattern in patterns.items():
        if meta["Type"] == "Full" and "LoRA" in hp_name:
            continue

        match = re.search(pattern, hp_string)
        if match:
            val_str = match.group(1).rstrip("-.")
            try:
                val = float(val_str)
                if hp_name in ["LoRA R", "LoRA A"]:
                    meta[hp_name] = int(val)
                elif hp_name == "LR":
                    meta[hp_name] = val
                elif val.is_integer():
                    meta[hp_name] = int(val)
                else:
                    meta[hp_name] = val
            except ValueError:
                meta[hp_name] = val_str
        elif hp_name not in meta:
            meta[hp_name] = np.nan

    return meta


def get_score_from_metrics(metrics_path, task_key):
    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
        if "all_primary_scores" in data:
            for score_str in data["all_primary_scores"]:
                parts = score_str.split()
                if len(parts) >= 2:
                    score_val = parts[-1]
                    name_part = " ".join(parts[:-1])

                    clean_task = task_key.replace("_tulu", "").replace("::tulu", "")
                    clean_metric_name = name_part.replace("_tulu", "")
                    if clean_task in clean_metric_name:
                        return float(score_val)
        return None
    except Exception:
        return None


def collect_data(results_dir, min_job_id=0, max_job_id=None):
    records = []
    if not os.path.exists(results_dir):
        return pd.DataFrame()

    tasks = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]

    for task in tasks:
        task_dir = os.path.join(results_dir, task)
        models = os.listdir(task_dir)

        for model in models:
            model_path = os.path.join(task_dir, model)
            if not os.path.isdir(model_path):
                continue

            meta = parse_model_name(model)

            # --- FILTERING ---
            if meta["Job ID"] < min_job_id:
                continue
            if max_job_id is not None and meta["Job ID"] > max_job_id:
                continue

            dates = sorted(
                [
                    d
                    for d in os.listdir(model_path)
                    if os.path.isdir(os.path.join(model_path, d))
                ]
            )
            if not dates:
                continue

            metrics_file = os.path.join(model_path, dates[0], "metrics.json")
            if os.path.exists(metrics_file):
                score = get_score_from_metrics(metrics_file, task)
                if score is not None:
                    record = meta.copy()
                    record["task"] = task.replace("_tulu", "").replace("::tulu", "")
                    record["score"] = score
                    records.append(record)

    return pd.DataFrame(records)


def format_latex_row(row_data, columns, is_best_dict=None):
    tex_parts = []

    hp_cols = ["Method", "Type", "LR", "Lambda", "Beta", "LoRA R", "LoRA A"]
    for col in hp_cols:
        if col in columns:
            val = row_data.get(col, "-")
            if pd.isna(val):
                tex_parts.append("-")
            elif col == "LR":
                tex_parts.append(f"{val:.0e}")
            elif col == "Method":
                tex_parts.append(str(val).title().replace("Deltaqwen", "Delta\_Qwen"))
            else:
                tex_parts.append(str(val))

    score_cols = [c for c in columns if c not in hp_cols and c != "Job ID"]

    for col in score_cols:
        val = row_data.get(col, np.nan)
        if pd.isna(val):
            tex_parts.append("-")
        else:
            val_str = f"{val:+.3f}"
            if is_best_dict and abs(val - is_best_dict.get(col, -999)) < 1e-9:
                val_str = f"\\textbf{{{val_str}}}"
            tex_parts.append(val_str)

    return " & ".join(tex_parts) + " \\\\"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--min_job_id", type=int, default=0)
    parser.add_argument("--max_job_id", type=int, default=None)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument(
        "--filter_negatives",
        action="store_true",
        help="Remove rows with any negative delta score",
    )
    parser.add_argument("--output_file", type=str, default="ablation_table.tex")
    args = parser.parse_args()

    df = collect_data(args.results_dir, args.min_job_id, args.max_job_id)

    if df.empty:
        print("No results found.")
        return

    # Pivot Data
    hp_cols = ["Method", "Type", "LR", "Lambda", "Beta", "LoRA R", "LoRA A"]
    active_hp_cols = [c for c in hp_cols if c in df.columns]

    pivot_df = df.pivot_table(
        index=["Job ID"] + active_hp_cols,
        columns="task",
        values="score",
        aggfunc="first",
    ).reset_index()

    # Calculate Deltas
    score_cols = [c for c in pivot_df.columns if c not in ["Job ID"] + active_hp_cols]
    valid_tasks = [t for t in score_cols if t in SFT_SCORES]

    if not valid_tasks:
        print("No matching tasks found for baseline subtraction.")
        return

    for task in valid_tasks:
        pivot_df[task] = pivot_df[task] - SFT_SCORES[task]

    # --- FILTER NEGATIVES ---
    if args.filter_negatives:
        initial_len = len(pivot_df)
        # Identify rows where ANY task score is negative (< 0)
        neg_mask = (pivot_df[valid_tasks] < 0).any(axis=1)
        pivot_df = pivot_df[~neg_mask]
        print(f"Filtered out {initial_len - len(pivot_df)} rows with negative scores.")

    # Mean & Sort
    pivot_df["Mean"] = pivot_df[valid_tasks].mean(axis=1)
    pivot_df = pivot_df.sort_values(by="Mean", ascending=False)

    top_df = pivot_df.head(args.top_n).copy()

    # Generate LaTeX
    display_cols = active_hp_cols + valid_tasks + ["Mean"]

    best_stats = {}
    if not top_df.empty:
        for col in valid_tasks + ["Mean"]:
            best_stats[col] = top_df[col].max()

    col_def = "l" * len(active_hp_cols) + "c" * (len(valid_tasks) + 1)

    with open(args.output_file, "w") as f:
        f.write(
            f"% Ablation Study: Top {args.top_n} (Filter Negatives: {args.filter_negatives})\n"
        )
        f.write("\\begin{table}[h]\n\\centering\n\\resizebox{\\textwidth}{!}{\n")
        f.write(f"\\begin{{tabular}}{{{col_def}}}\n")
        f.write("\\toprule\n")

        headers = []
        for c in display_cols:
            if c == "Lambda":
                headers.append(r"$\lambda$")
            elif c == "Beta":
                headers.append(r"$\beta$")
            elif c == "LoRA R":
                headers.append(r"$r$")
            elif c == "LoRA A":
                headers.append(r"$\alpha$")
            else:
                headers.append(c.title())

        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")

        for _, row in top_df.iterrows():
            f.write(format_latex_row(row, display_cols, best_stats) + "\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n}\n")
        f.write(f"\\caption{{Ablation Study: Top {args.top_n} Configurations}}\n")
        f.write("\\end{table}\n")

    print(f"Ablation table saved to {args.output_file}")


if __name__ == "__main__":
    main()
