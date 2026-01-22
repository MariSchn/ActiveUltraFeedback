import os
import json
import argparse
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure plot style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

# ==============================================================================
#                             CONFIGURATION
# ==============================================================================

TASK_NAME_MAP = {
    "gsm8k::tulu": "GSM8K",
    "ifeval::tulu": "IF Eval",
    "minerva_math::tulu": "Minerva Math",
    "truthfulqa::tulu": "Truthful QA",
    "gsm8k": "GSM8K",
    "ifeval": "IF Eval",
    "minerva_math": "Minerva Math",
    "truthfulqa": "Truthful QA",
}

# UPDATED BASELINES (Includes SFT for x=0 anchor)
TASK_BASELINES = {
    "GSM8K": {
        "SFT": 0.758,
        "DeltaQwen": 0.813,
        "MaxMin": 0.780,
        "Random": 0.782,
        "UltraFeedback": 0.795,
    },
    "IF Eval": {
        "SFT": 0.713,
        "DeltaQwen": 0.760,
        "MaxMin": 0.697,
        "Random": 0.741,
        "UltraFeedback": 0.712,
    },
    "Minerva Math": {
        "SFT": 0.309,
        "DeltaQwen": 0.380,
        "MaxMin": 0.388,
        "Random": 0.335,
        "UltraFeedback": 0.346,
    },
    "Truthful QA": {
        "SFT": 0.468,
        "DeltaQwen": 0.598,
        "MaxMin": 0.618,
        "Random": 0.524,
        "UltraFeedback": 0.507,
    },
}

MEAN_BASELINES = {
    "SFT": 0.562,
    "DeltaQwen": 0.638,
    "MaxMin": 0.621,
    "Random": 0.596,
    "UltraFeedback": 0.590,
}

# ==============================================================================
#                                   HELPERS
# ==============================================================================


def extract_info_from_path(file_path):
    """
    Walks up the path to find 'ID-Method_Size' pattern.
    """
    path_parts = os.path.normpath(file_path).split(os.sep)
    for part in reversed(path_parts):
        # Matches: digits + hyphen + method + underscore + digits
        # Example: 01-MyMethod_1000
        match = re.match(r"^\d+-(.+)_(\d+)$", part)
        if match:
            return match.group(1), int(match.group(2))

        # Fallback for simple "method_size" without ID
        # Example: MyMethod_1000
        match_simple = re.match(r"^(.+)_(\d+)$", part)
        if match_simple and not match_simple.group(1).isdigit():
            return match_simple.group(1), int(match_simple.group(2))

    return None, None


def parse_metrics_file(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        if "all_primary_scores" in data and isinstance(
            data["all_primary_scores"], list
        ):
            raw_entry = data["all_primary_scores"][0]
            if ":" in raw_entry:
                parts = raw_entry.rsplit(":", 1)
                clean_name = TASK_NAME_MAP.get(parts[0].strip(), parts[0].strip())
                return clean_name, float(parts[1].strip())
    except Exception:
        return None, None
    return None, None


def collect_data_from_single_dir(root_dir):
    """
    Collects data recursively from a single root directory.
    """
    records = []
    # Check if dir exists
    if not os.path.exists(root_dir):
        print(f"Warning: Directory not found: {root_dir}")
        return pd.DataFrame()

    files = glob.glob(os.path.join(root_dir, "**", "metrics.json"), recursive=True)
    print(f"Scanning {root_dir}... Found {len(files)} metrics files.")

    for file_path in files:
        method, size = extract_info_from_path(file_path)
        # Skip if folder naming convention isn't met
        if size is None:
            continue

        task, score = parse_metrics_file(file_path)
        if task and score is not None:
            records.append(
                {
                    "Method": method,
                    "Size": size,
                    "Task": task,
                    "Score": score,
                    "Source": root_dir,
                }
            )

    return pd.DataFrame(records)


def set_linear_xaxis(df, plt_obj):
    max_val = df["Size"].max()
    upper_bound = int(max_val) + 5000
    ticks = np.arange(0, upper_bound, 5000)
    plt_obj.xticks(ticks)
    plt_obj.xlim(0, upper_bound)


def draw_baselines(plt_obj, baselines, x_pos_text):
    colors = sns.color_palette("tab10", len(baselines))
    for i, (name, val) in enumerate(baselines.items()):
        plt_obj.axhline(y=val, color=colors[i], linestyle="--", alpha=0.6, linewidth=1)
        plt_obj.text(
            x=x_pos_text,
            y=val,
            s=name,
            color=colors[i],
            fontweight="bold",
            fontsize=8,
            ha="right",
            va="bottom",
        )


# ==============================================================================
#                                 PLOTTING
# ==============================================================================


def plot_single_task(df, task_name, output_dir, color_map, show_baselines=True):
    subset = df[df["Task"] == task_name].copy().sort_values(by="Size")
    if subset.empty:
        return

    baselines = TASK_BASELINES.get(task_name, {})

    # Add SFT (Size 0) points so all lines start at the same origin
    if "SFT" in baselines:
        sft_score = baselines["SFT"]
        start_points = [
            {"Size": 0, "Task": task_name, "Score": sft_score, "Method": m}
            for m in subset["Method"].unique()
        ]
        subset = pd.concat([pd.DataFrame(start_points), subset], ignore_index=True)

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=subset,
        x="Size",
        y="Score",
        hue="Method",
        marker="o",
        linewidth=2.5,
        palette=color_map,
    )

    set_linear_xaxis(subset, plt)
    if show_baselines:
        draw_baselines(plt, baselines, subset["Size"].max())

    plt.title(f"{task_name}: Method Comparison", fontsize=16)
    plt.xlabel("Training Samples", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(title="Strategy")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()

    safe_name = task_name.replace(" ", "_").lower()
    plt.savefig(os.path.join(output_dir, f"dpo_compare_{safe_name}.png"), dpi=300)
    plt.close()


def plot_mean_score(df, output_dir, color_map, show_baselines=True):
    if df.empty:
        return
    mean_df = df.groupby(["Method", "Size"])["Score"].mean().reset_index()

    if "SFT" in MEAN_BASELINES:
        sft_score = MEAN_BASELINES["SFT"]
        start_points = [
            {"Size": 0, "Score": sft_score, "Method": m}
            for m in mean_df["Method"].unique()
        ]
        mean_df = pd.concat([pd.DataFrame(start_points), mean_df], ignore_index=True)

    plt.figure(figsize=(12, 7))

    sns.lineplot(
        data=mean_df,
        x="Size",
        y="Score",
        hue="Method",
        marker="s",
        linewidth=3,
        palette=color_map,
    )

    set_linear_xaxis(mean_df, plt)
    if show_baselines:
        draw_baselines(plt, MEAN_BASELINES, mean_df["Size"].max())

    plt.title("Mean Score (Avg across all tasks)", fontsize=16)
    plt.xlabel("Training Samples", fontsize=14)
    plt.ylabel("Mean Score", fontsize=14)
    plt.legend(title="Strategy")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dpo_mean_comparison.png"), dpi=300)
    plt.close()


# ==============================================================================
#                                   MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Changed from single string to nargs='+' to accept multiple folders
    parser.add_argument(
        "--results_dirs",
        nargs="+",
        required=True,
        help="List of result folders to aggregate",
    )
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--hide_baselines", action="store_true")
    args = parser.parse_args()

    all_data_frames = []

    print(f"Aggregating data from {len(args.results_dirs)} directories...")

    # Iterate through each directory provided in arguments
    for result_dir in args.results_dirs:
        df_part = collect_data_from_single_dir(result_dir)
        if not df_part.empty:
            all_data_frames.append(df_part)

    if not all_data_frames:
        print("No valid data found in any of the provided directories.")
        exit(0)

    # Combine all individual dataframes
    df = pd.concat(all_data_frames, ignore_index=True)
    df = df.sort_values(by="Size")

    # --- CREATE CONSISTENT COLOR MAP ---
    # Get all unique methods from the aggregated dataframe
    unique_methods = sorted(df["Method"].unique())

    # Create a palette with enough distinct colors
    palette = sns.color_palette("bright", n_colors=len(unique_methods))

    # Map each method name to a specific color
    method_color_map = dict(zip(unique_methods, palette))

    print("\n--- Summary ---")
    print(f"Sources: {[d for d in args.results_dirs]}")
    print(f"Methods: {unique_methods}")
    print(f"Tasks:   {df['Task'].unique()}")
    print("---------------\n")

    os.makedirs(args.output_dir, exist_ok=True)
    show_baselines = not args.hide_baselines

    # Pass method_color_map to functions
    plot_mean_score(df, args.output_dir, method_color_map, show_baselines)

    for task in df["Task"].unique():
        plot_single_task(df, task, args.output_dir, method_color_map, show_baselines)

    print(f"Plots saved to: {os.path.abspath(args.output_dir)}")
