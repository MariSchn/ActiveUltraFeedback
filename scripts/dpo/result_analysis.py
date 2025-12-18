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

# --- CONFIGURATION: NAMING & ORDERING ---
ACQ_NAME_MAP = {
    "random": "Random",
    "ultrafeedback": "UltraFeedback",
    "maxmin": "MaxMin",
    "delta": "Delta\_Qwen",
    "deltaqwen": "Delta\_Qwen",
    "drts": "DRTS",
    "deltaucb": "DeltaUCB",
    "deltaquantile": "DeltaQuantile",
    "maxminlcb": "MaxMinLCB",
    "doublets": "DoubleTS",
    "infogain": "InfoGain",
    "infomax": "InfoMax",
    "batch": "Batch",
    "base": "Base",
}

ORDER_LIST = [
    "Random",
    "UltraFeedback",
    "MaxMin",
    "Delta\_Qwen",
    # --- MIDRULE CUTOFF ---
    "DRTS",
    "DeltaUCB",
    "DeltaQuantile",
    "MaxMinLCB",
    "DoubleTS",
    "InfoMax",
    "InfoGain",
]

ORDER_RANK = {name: i for i, name in enumerate(ORDER_LIST)}
MIDRULE_AFTER_INDEX = ORDER_LIST.index("Delta\_Qwen")


def get_display_name_base(raw_name):
    norm = raw_name.lower().replace("_", "").replace("-", "")
    if norm in ACQ_NAME_MAP:
        return ACQ_NAME_MAP[norm]
    for key, val in ACQ_NAME_MAP.items():
        if key in norm:
            return val
    return raw_name.title()


def parse_model_name(name):
    meta = {}

    # 1. Job ID
    match_job = re.match(r"^(\d+)-", name)
    if match_job:
        meta["Job ID"] = int(match_job.group(1))
        rest = name[len(match_job.group(0)) :]
    else:
        meta["Job ID"] = -1
        rest = name

    # 2. Acquisition Function
    parts = rest.split("-")
    acq_block = parts[0]
    raw_acq = acq_block.split("_")[0] if "_" in acq_block else acq_block
    meta["Acq Func"] = get_display_name_base(raw_acq)

    # 3. Hyperparameters
    hp_string = "-".join(parts[1:])
    patterns = {
        "LR": r"lr([0-9\.eE-]+)",
        "SG": r"sg([\d\.]+)",
        "Beta": r"b([\d\.]+)",
        "LoRA R": r"loraR(\d+)",
        "LoRA A": r"loraA(\d+)",
    }

    for hp_name, pattern in patterns.items():
        match = re.search(pattern, hp_string)
        if match:
            # Strip trailing chars (like - or .) to safely convert numbers like 2.5e-6
            val_str = match.group(1).rstrip("-.")
            try:
                val = float(val_str)
                if hp_name == "LR":
                    meta[hp_name] = val
                elif val.is_integer():
                    meta[hp_name] = int(val)
                else:
                    meta[hp_name] = val
            except ValueError:
                meta[hp_name] = val_str
        else:
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


def collect_data(results_dir, min_job_id=0):
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
            if meta["Job ID"] < min_job_id:
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
                    raw_task = task.replace("_tulu", "").replace("::tulu", "")
                    record["task"] = raw_task
                    record["score"] = score
                    records.append(record)

    return pd.DataFrame(records)


def format_latex_row(name, scores, is_best_mask=None, is_delta=False):
    row_str = f"{name}"
    for i, score in enumerate(scores):
        val_str = "-"
        if pd.isna(score):
            val_str = "-"
        else:
            if is_delta:
                val_str = f"{score:+.3f}"
            else:
                val_str = f"{score:.3f}"

        if is_best_mask and is_best_mask[i] and not pd.isna(score):
            val_str = f"\\textbf{{{val_str}}}"

        row_str += f" & {val_str}"
    row_str += " \\\\"
    return row_str


def generate_latex_table(df, sft_scores):
    meta_cols = [
        "Job ID",
        "Acq Func",
        "LR",
        "SG",
        "Beta",
        "LoRA R",
        "LoRA A",
        "Average",
        "Display Name",
        "SortRank",
        "InstanceRank",
    ]
    potential_tasks = [
        c for c in df.columns if c in sft_scores.keys() or c not in meta_cols
    ]
    tasks = sorted([t for t in potential_tasks if t in df.columns])

    if not tasks:
        return "No matching tasks found."

    # Delta Calculation
    delta_df = df.copy()
    for task in tasks:
        if task in sft_scores:
            delta_df[task] = delta_df[task] - sft_scores[task]
    delta_df["Mean"] = delta_df[tasks].mean(axis=1)

    # Ranking
    delta_df["SortRank"] = delta_df["Acq Func"].map(lambda x: ORDER_RANK.get(x, 999))

    # Display Name
    def make_name(row):
        hps = []
        if pd.notna(row.get("Beta")):
            hps.append(f"$\\beta={row['Beta']}$")
        if pd.notna(row.get("SG")):
            hps.append(f"$sg={row['SG']}$")

        hp_str = ", ".join(hps)
        name = row["Acq Func"]
        if hp_str:
            return f"{name} ({hp_str})"
        return name

    delta_df["Display Name"] = delta_df.apply(make_name, axis=1)

    # Global Bests
    best_deltas = {}
    for task in tasks + ["Mean"]:
        best_deltas[task] = delta_df[task].max()

    # LaTeX Construction
    col_def = "l" + "c" * (len(tasks) + 1)
    latex = []
    latex.append(f"\\begin{{tabular}}{{{col_def}}}")
    latex.append("\\toprule")
    headers = [t.replace("_", " ").title() for t in tasks] + ["Mean"]
    latex.append(f" & {' & '.join(headers)} \\\\")
    latex.append("\\midrule")

    # SFT Row
    sft_vals = [sft_scores.get(t, np.nan) for t in tasks]
    sft_mean = np.nanmean(sft_vals)
    latex.append(
        format_latex_row("SFT Base Model", sft_vals + [sft_mean], is_delta=False)
    )
    latex.append("\\midrule")
    latex.append("\\midrule")

    # Sort Rows
    delta_df = delta_df.sort_values(by=["SortRank", "Job ID"], ascending=[True, False])

    midrule_inserted = False

    for idx, row in delta_df.iterrows():
        current_rank = row["SortRank"]

        if not midrule_inserted and current_rank > MIDRULE_AFTER_INDEX:
            latex.append("\\midrule")
            midrule_inserted = True

        vals = [row[t] for t in tasks] + [row["Mean"]]
        is_best = []
        for i, task_name in enumerate(tasks + ["Mean"]):
            is_best.append(abs(vals[i] - best_deltas[task_name]) < 1e-9)

        latex.append(
            format_latex_row(
                row["Display Name"], vals, is_best_mask=is_best, is_delta=True
            )
        )

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")

    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--min_job_id", type=int, default=0)
    parser.add_argument("--output_file", type=str, default="results_final.tex")
    args = parser.parse_args()

    df = collect_data(args.results_dir, args.min_job_id)

    if df.empty:
        print("No results found.")
        return

    # Pivot Data
    hp_cols = ["LR", "SG", "Beta", "LoRA R", "LoRA A"]
    active_hp_cols = [c for c in hp_cols if c in df.columns]

    pivot_df = df.pivot_table(
        index=["Job ID", "Acq Func"] + active_hp_cols,
        columns="task",
        values="score",
        aggfunc="first",
    ).reset_index()

    if "LR" in pivot_df.columns:
        pivot_df["LR"] = pd.to_numeric(pivot_df["LR"], errors="coerce")
        pivot_df = pivot_df.dropna(subset=["LR"])

    with open(args.output_file, "w") as f:
        f.write("% Results Table (Ranked by Job ID)\n\n")

        if "LR" in pivot_df.columns:
            unique_lrs = sorted(pivot_df["LR"].unique())

            for lr in unique_lrs:
                lr_data = pivot_df[pivot_df["LR"] == lr].copy()

                # --- RANKING LOGIC ---
                lr_data = lr_data.sort_values(by="Job ID")
                lr_data["InstanceRank"] = lr_data.groupby("Acq Func").cumcount()
                max_rank = lr_data["InstanceRank"].max()

                for r in range(max_rank + 1):
                    batch_df = lr_data[lr_data["InstanceRank"] == r]
                    if not batch_df.empty:
                        f.write(
                            "\\begin{table}[h]\n\\centering\n\\resizebox{\\textwidth}{!}{\n"
                        )
                        f.write(generate_latex_table(batch_df, SFT_SCORES))
                        f.write("\n}\n")
                        # UPDATED CAPTION: Uses .1e to preserve 2.5e-06
                        f.write(f"\\caption{{Learning Rate {lr:.2e}}}\n")
                        f.write("\\end{table}\n\n")

                f.write("\\clearpage\n")

        else:
            f.write(generate_latex_table(pivot_df, SFT_SCORES))

    print(f"Report generated: {args.output_file}")


if __name__ == "__main__":
    main()
