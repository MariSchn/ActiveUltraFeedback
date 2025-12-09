import os
import json
import argparse
import pandas as pd
import numpy as np
import glob

# ==============================================================================
#                             CONFIGURATION
# ==============================================================================

# Columns to display in the table
CPO_COLS = ["GSM8K", "IF Eval", "Minerva Math", "Truthful QA", "Mean"]

# Mapping folder names (like 'gsm8k_tulu') to Table Columns
TASK_MAP = {
    "gsm8k": "GSM8K",
    "ifeval": "IF Eval",
    "minerva_math": "Minerva Math",
    "truthfulqa": "Truthful QA",
}

# ABSOLUTE BASELINE (SFT Base Model) - Used to calc deltas
SFT_BASE = {
    "GSM8K": 0.758,
    "IF Eval": 0.713,
    "Minerva Math": 0.309,
    "Truthful QA": 0.468,
    "Mean": 0.562,
}

# ==============================================================================
#                                   HELPERS
# ==============================================================================


def format_delta(val, is_best=False):
    """
    Formats a float delta into LaTeX.
    - None/NaN -> "-"
    - Positive -> "+0.123"
    - Negative -> "-0.123"
    - Best -> \textbf{}
    """
    if val is None or pd.isna(val):
        return "-"

    sign = "+" if val >= 0 else ""
    text = f"{sign}{val:.3f}"

    if is_best:
        return f"\\textbf{{{text}}}"
    return text


def format_sft(val):
    """Formats absolute score for SFT row."""
    return f"{val:.3f}"


def extract_primary_score(json_path):
    """Reads the 'primary_score' from the metrics.json."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Format 1: {"metrics": [{"primary_score": 0.123, ...}]}
        if "metrics" in data and isinstance(data["metrics"], list):
            return float(data["metrics"][0].get("primary_score"))

        # Format 2: {"primary_score": 0.123}
        if "primary_score" in data:
            return float(data["primary_score"])

    except Exception:
        return None
    return None


def collect_local_data(root_dir):
    """
    Scans root_dir recursively.
    It expects structure like: root/task_name/run_id/timestamp/metrics.json
    """
    data = {}  # {run_id: {task: score}}

    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} does not exist.")
        return pd.DataFrame()

    # Find ALL metrics.json files recursively
    print(f"Scanning {root_dir}...")
    metric_files = glob.glob(
        os.path.join(root_dir, "**", "metrics.json"), recursive=True
    )
    print(f"Found {len(metric_files)} metrics files.")

    for fpath in metric_files:
        # Example path: .../cpores/ifeval_tulu/1201378-random_60829/2025-12-07-15-36/metrics.json
        parts = fpath.split(os.sep)

        # We need to identify the run_id (e.g., 1201378-random_60829)
        # and the task (e.g., ifeval_tulu).
        # Heuristic: usually run_id is the parent of the timestamp folder, or 2 levels up from json

        # Assuming standard structure: task/run_id/timestamp/metrics.json
        if len(parts) < 4:
            continue

        run_id = parts[-3]  # 1201378-random_60829
        task_folder = parts[-4]  # ifeval_tulu

        # Normalize task name
        task_col = None
        for key, col_name in TASK_MAP.items():
            if key in task_folder.lower():
                task_col = col_name
                break

        if not task_col:
            continue

        val = extract_primary_score(fpath)
        if val is not None:
            # Safe name for LaTeX
            display_name = run_id.replace("_", "\\_")

            if display_name not in data:
                data[display_name] = {col: np.nan for col in CPO_COLS if col != "Mean"}

            # Calculate Delta
            data[display_name][task_col] = val - SFT_BASE[task_col]

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient="index")

    if not df.empty:
        # Calculate Mean row-wise (ignoring NaNs)
        df["Mean"] = df.mean(axis=1, skipna=True)

        # Ensure all columns exist
        for col in CPO_COLS:
            if col not in df.columns:
                df[col] = np.nan

        # Reorder
        df = df[CPO_COLS]
        # Sort by name
        df = df.sort_index()

    return df


def process_dataframe_formatting(df):
    """Finds max per column and formats strings."""
    if df.empty:
        return pd.DataFrame()

    df_fmt = df.copy()
    # Calculate max for bolding logic
    max_series = df.max(numeric_only=True)

    for col in df.columns:
        max_val = max_series[col]
        for idx in df.index:
            val = df.at[idx, col]
            # Bold if it's the max (and not NaN)
            is_best = (val == max_val) and (not pd.isna(val))
            df_fmt.at[idx, col] = format_delta(val, is_best)

    return df_fmt


def get_static_data():
    # SFT Row (Absolute scores)
    df_sft = pd.DataFrame([SFT_BASE], index=["SFT Base Model"])
    df_sft = df_sft[CPO_COLS]
    for c in df_sft.columns:
        df_sft[c] = df_sft[c].apply(format_sft)
    return df_sft


def write_latex(filename, df_sft, df_local_fmt):
    with open(filename, "w") as f:
        f.write("% Required: \\usepackage{booktabs}, \\usepackage{graphicx}\n")

        col_fmt = "l" + "c" * len(CPO_COLS)
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write(f"\\begin{{tabular}}{{{col_fmt}}}\n")
        f.write("\\toprule\n")

        headers = ["Method"] + CPO_COLS
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")

        # 1. SFT (Absolute)
        f.write(f"SFT Base Model & {' & '.join(df_sft.iloc[0].values)} \\\\\n")
        f.write("\\midrule\n")

        # 2. Local Results
        if not df_local_fmt.empty:
            for idx, row in df_local_fmt.iterrows():
                f.write(f"{idx} & {' & '.join(row.values)} \\\\\n")
        else:
            f.write(f"% No local runs found \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write(
            "\\caption{CPO Evaluation Results (SFT: Absolute, Runs: Deltas vs SFT). '-' indicates missing eval.}\n"
        )
        f.write("\\end{table}\n")


# ==============================================================================
#                                   MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to 'cpores' or 'tmpres' directory",
    )
    parser.add_argument(
        "--output", type=str, default="cpo_local_table.tex", help="Output LaTeX file"
    )
    args = parser.parse_args()

    # 1. Get Static SFT
    df_sft = get_static_data()

    # 2. Collect Local Data
    df_local_num = collect_local_data(args.results_dir)

    # 3. Format Data
    df_local_fmt = process_dataframe_formatting(df_local_num)

    # 4. Save
    write_latex(args.output, df_sft, df_local_fmt)
    print(f"Table generated at: {os.path.abspath(args.output)}")
