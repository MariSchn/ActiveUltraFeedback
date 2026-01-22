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

    # 4. Extract Hyperparameters (Including Seed)
    hp_string = "-".join(parts[1:])
    patterns = {
        "LR": r"lr([0-9\.eE-]+)",
        "Lambda": r"sg([\d\.]+)",
        "Beta": r"b([\d\.]+)",
        "LoRA R": r"loraR(\d+)",
        "LoRA A": r"loraA(\d+)",
        "Seed": r"seed(\d+)",
    }

    for hp_name, pattern in patterns.items():
        if meta["Type"] == "Full" and "LoRA" in hp_name:
            continue

        match = re.search(pattern, hp_string)
        if match:
            val_str = match.group(1).rstrip("-.")
            try:
                val = float(val_str)
                if hp_name in ["LoRA R", "LoRA A", "Seed"]:
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


def write_latex_table(
    f, df, display_cols, caption, is_best_dict=None, is_summary=False
):
    # Calculate column alignment (l for HPs, c for metrics)
    col_def = "l" * (len(display_cols) - 4) + "c" * 4
    if len(display_cols) < 4:
        col_def = "l" * len(display_cols)  # fallback

    f.write("\\begin{table}[h]\n\\centering\n\\resizebox{\\textwidth}{!}{\n")
    f.write(f"\\begin{{tabular}}{{{col_def}}}\n")
    f.write("\\toprule\n")

    # --- Header ---
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

    # --- Rows ---
    for _, row in df.iterrows():
        tex_parts = []
        for col in display_cols:
            val = row.get(col, "-")

            # Formatting based on column type
            if col in ["Method", "Type", "Job ID", "Seed"]:
                if col == "Method":
                    tex_parts.append(
                        str(val).title().replace("Deltaqwen", "Delta\_Qwen")
                    )
                else:
                    tex_parts.append(str(val))
            elif col in ["LR", "Lambda", "Beta", "LoRA R", "LoRA A"]:
                if col == "LR" and isinstance(val, (int, float)):
                    tex_parts.append(f"{val:.0e}")
                elif pd.isna(val):
                    tex_parts.append("-")
                else:
                    tex_parts.append(str(val))
            else:
                # Score Columns
                if is_summary:
                    # Expecting a pre-formatted string "Mean \pm Std"
                    tex_parts.append(str(val))
                else:
                    if pd.isna(val):
                        tex_parts.append("-")
                    else:
                        # Float formatting with optional Bold for best score
                        val_str = f"{val:+.3f}"
                        if (
                            is_best_dict
                            and isinstance(val, (int, float))
                            and abs(val - is_best_dict.get(col, -999)) < 1e-9
                        ):
                            val_str = f"\\textbf{{{val_str}}}"
                        tex_parts.append(val_str)

        f.write(" & ".join(tex_parts) + " \\\\\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n}\n")
    f.write(f"\\caption{{{caption}}}\n")
    f.write("\\end{table}\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--min_job_id", type=int, default=0)
    parser.add_argument("--max_job_id", type=int, default=None)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print all results sorted by hyperparameters (Standard Mode)",
    )
    parser.add_argument(
        "--filter_negatives",
        action="store_true",
        help="Remove rows with any negative delta score",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Enable Summary Mode: Prints Individual Run table AND Aggregated Mean/Std table",
    )
    parser.add_argument("--output_file", type=str, default="ablation_table.tex")
    args = parser.parse_args()

    df = collect_data(args.results_dir, args.min_job_id, args.max_job_id)

    if df.empty:
        print("No results found.")
        return

    # Pivot Data
    hp_cols = ["Method", "Type", "LR", "Lambda", "Beta", "LoRA R", "LoRA A", "Seed"]
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

    # Filter Negatives (Common to all modes)
    if args.filter_negatives:
        initial_len = len(pivot_df)
        neg_mask = (pivot_df[valid_tasks] < 0).any(axis=1)
        pivot_df = pivot_df[~neg_mask]
        print(f"Filtered out {initial_len - len(pivot_df)} rows with negative scores.")

    # Calculate Mean
    pivot_df["Mean"] = pivot_df[valid_tasks].mean(axis=1)

    # ================= MODE SELECTION =================
    with open(args.output_file, "w") as f:
        f.write("% Auto-generated Tables\n")

        if args.summary:
            # ---------------- SUMMARY MODE ----------------
            print("Mode: Summary (Aggregating Seeds)")

            # Table 1: Individual Rows (Sorted by HP then Seed)
            sort_cols = [c for c in active_hp_cols if c != "Seed"]
            if "Seed" in active_hp_cols:
                sort_cols.append("Seed")

            table1_df = pivot_df.sort_values(by=sort_cols, ascending=True)
            display_cols_1 = active_hp_cols + valid_tasks + ["Mean"]

            write_latex_table(
                f,
                table1_df,
                display_cols_1,
                "Individual Experimental Results (All Seeds)",
            )

            # Table 2: Aggregated Summary
            group_cols = [c for c in active_hp_cols if c not in ["Seed", "Job ID"]]
            if not group_cols:
                print(
                    "Warning: Cannot group (no hyperparameters found). Skipping summary table."
                )
            else:
                target_cols = valid_tasks + ["Mean"]
                agg_df = (
                    pivot_df.groupby(group_cols)[target_cols]
                    .agg(["mean", "std"])
                    .reset_index()
                )

                # Format as "Mean +/- Std"
                summary_rows = []
                for _, row in agg_df.iterrows():
                    new_row = {}
                    for hp in group_cols:
                        new_row[hp] = row[(hp, "")]
                    for task in target_cols:
                        m = row[(task, "mean")]
                        s = row[(task, "std")]
                        new_row[task] = f"{m:+.3f}_{{ \\pm {s:.3f} }}"
                    summary_rows.append(new_row)

                summary_df = pd.DataFrame(summary_rows)
                display_cols_2 = group_cols + valid_tasks + ["Mean"]

                write_latex_table(
                    f,
                    summary_df,
                    display_cols_2,
                    "Aggregated Results (Mean $\\pm$ Std Dev)",
                    is_summary=True,
                )

        else:
            # ---------------- STANDARD MODE ----------------
            print("Mode: Standard (Top N / All)")

            if args.all:
                # Sort by Hyperparameters
                final_df = pivot_df.sort_values(by=active_hp_cols, ascending=True)
                caption = "Ablation Study: All Configurations"
            else:
                # Sort by Score (Top N)
                pivot_df = pivot_df.sort_values(by="Mean", ascending=False)
                final_df = pivot_df.head(args.top_n).copy()
                caption = f"Ablation Study: Top {args.top_n} Configurations"

            # Calculate Best Stats for Bolding
            best_stats = {}
            if not final_df.empty:
                for col in valid_tasks + ["Mean"]:
                    best_stats[col] = final_df[col].max()

            display_cols = active_hp_cols + valid_tasks + ["Mean"]
            write_latex_table(
                f, final_df, display_cols, caption, is_best_dict=best_stats
            )

    print(f"Tables saved to {args.output_file}")


if __name__ == "__main__":
    main()
