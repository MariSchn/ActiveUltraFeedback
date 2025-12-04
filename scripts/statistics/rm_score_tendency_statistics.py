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
#                                CONFIGURATION
# ==============================================================================

# 1. BASELINES FOR MEAN SCORE
RM_MEAN_BASELINES = {
    "SFT": 0.290,
    "Random": 0.554,
    "UltraFeedback": 0.546,
    "MaxMin": 0.593,
    "DeltaQwen": 0.383,
}

# 2. BASELINES FOR INDIVIDUAL TASKS
# SFT Base + Deltas provided in prompt
RM_TASK_BASELINES = {
    "Factuality": {
        "SFT": 0.316,
        "Random": 0.316 + 0.409,  # 0.725
        "UltraFeedback": 0.316 + 0.392,  # 0.708
        "MaxMin": 0.316 + 0.373,  # 0.689
        "DeltaQwen": 0.316 + 0.183,  # 0.499
    },
    "Focus": {
        "SFT": 0.277,
        "Random": 0.277 + 0.252,  # 0.529
        "UltraFeedback": 0.277 + 0.183,  # 0.460
        "MaxMin": 0.277 + 0.392,  # 0.669
        "DeltaQwen": 0.277 - 0.050,  # 0.227
    },
    "Math": {
        "SFT": 0.445,
        "Random": 0.445 + 0.164,  # 0.609
        "UltraFeedback": 0.445 + 0.200,  # 0.645
        "MaxMin": 0.445 + 0.156,  # 0.601
        "DeltaQwen": 0.445 + 0.030,  # 0.475
    },
    "Precise IF": {
        "SFT": 0.261,
        "Random": 0.261 + 0.104,  # 0.365
        "UltraFeedback": 0.261 + 0.064,  # 0.325
        "MaxMin": 0.261 + 0.164,  # 0.425
        "DeltaQwen": 0.261 + 0.077,  # 0.338
    },
    "Safety": {
        "SFT": 0.347,
        "Random": 0.347 + 0.422,  # 0.769
        "UltraFeedback": 0.347 + 0.405,  # 0.752
        "MaxMin": 0.347 + 0.367,  # 0.714
        "DeltaQwen": 0.347 + 0.209,  # 0.556
    },
    "Ties": {
        "SFT": 0.095,
        "Random": 0.095 + 0.228,  # 0.323
        "UltraFeedback": 0.095 + 0.288,  # 0.383
        "MaxMin": 0.095 + 0.366,  # 0.461
        "DeltaQwen": 0.095 + 0.109,  # 0.204
    },
}

# ==============================================================================
#                                   HELPERS
# ==============================================================================


def extract_size_from_path(file_path):
    """Extracts dataset size (e.g., _10000) from path."""
    path_parts = os.path.normpath(file_path).split(os.sep)
    for part in reversed(path_parts):
        match = re.search(r"_(\d+)$", part)
        if match:
            return int(match.group(1))
    return None


def parse_metrics_file(file_path):
    """
    Parses metrics.json and returns a dictionary of all scores.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data  # Returns dict like {"Factuality": 0.7, "Mean": 0.5...}
    except Exception as e:
        print(f"[Warn] Error parsing {file_path}: {e}")
        return None


def collect_data(root_dir):
    records = []
    search_pattern = os.path.join(root_dir, "**", "metrics.json")
    files = glob.glob(search_pattern, recursive=True)

    print(f"Found {len(files)} metrics files. Parsing...")

    for file_path in files:
        size = extract_size_from_path(file_path)
        if size is None:
            continue

        data = parse_metrics_file(file_path)
        if data:
            # Iterate over all keys in the json (Factuality, Mean, etc.)
            for task_name, score in data.items():
                # Filter out non-numeric values if any
                if isinstance(score, (int, float)):
                    records.append(
                        {"Size": size, "Task": task_name, "Score": float(score)}
                    )

    return pd.DataFrame(records)


def set_linear_xaxis(df):
    """Sets x-axis to linear scale with ticks every 5000."""
    max_val = df["Size"].max()
    upper_bound = int(max_val) + 5000
    ticks = np.arange(0, upper_bound, 5000)
    plt.xticks(ticks)
    plt.xlim(0, upper_bound)
    return upper_bound


def draw_baselines(plt_obj, baselines, x_pos_text):
    """Helper to draw horizontal lines and labels."""
    colors = sns.color_palette("tab10", len(baselines))
    for i, (name, val) in enumerate(baselines.items()):
        plt_obj.axhline(y=val, color=colors[i], linestyle="--", alpha=0.7)
        plt_obj.text(
            x=x_pos_text,
            y=val + 0.002,
            s=f"{name}",
            color=colors[i],
            fontweight="bold",
            fontsize=9,
            ha="right",
            va="bottom",
        )


# ==============================================================================
#                                PLOTTING
# ==============================================================================


def plot_single_task(df, task_name, output_dir, baselines):
    """Plots a single task (or Mean) with baselines and SFT connection."""

    # Filter data for this task
    subset = df[df["Task"] == task_name].copy().sort_values(by="Size")

    if subset.empty:
        print(f"Skipping plot for {task_name} (No data)")
        return

    # --- CONNECT TO 0 (SFT) ---
    if baselines and "SFT" in baselines:
        sft_score = baselines["SFT"]
        start_point = pd.DataFrame([{"Size": 0, "Task": task_name, "Score": sft_score}])
        subset = pd.concat([start_point, subset], ignore_index=True)

    plt.figure(figsize=(10, 6))

    # Choose color based on whether it is "Mean" or a sub-task
    line_color = "firebrick" if task_name == "Mean" else None

    sns.lineplot(
        data=subset,
        x="Size",
        y="Score",
        marker="o",
        color=line_color,
        linewidth=2.5,
        label=task_name,
    )

    set_linear_xaxis(subset)

    if baselines:
        draw_baselines(plt, baselines, subset["Size"].max())

    plt.title(f"RM {task_name} Score vs. Dataset Size", fontsize=16)
    plt.xlabel("Number of Training Samples", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()

    # Safe filename
    safe_name = task_name.replace(" ", "_").lower()
    save_path = os.path.join(output_dir, f"rm_score_{safe_name}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved {task_name} plot to: {save_path}")
    plt.close()


def plot_combined_tasks(df, output_dir):
    """Plots all tasks on one chart (excluding Mean to avoid skewing scale)."""
    # Exclude Mean for the combined view as it's an aggregate
    subset = df[df["Task"] != "Mean"].copy().sort_values(by="Size")

    if subset.empty:
        return

    plt.figure(figsize=(14, 8))

    sns.lineplot(
        data=subset, x="Size", y="Score", hue="Task", marker="o", linewidth=2.5
    )

    set_linear_xaxis(subset)

    plt.title("All RM Tasks: Scores vs. Dataset Size", fontsize=16)
    plt.xlabel("Number of Training Samples", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "rm_all_tasks_combined.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved combined plot to: {save_path}")
    plt.close()


# ==============================================================================
#                                   MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RM evaluation results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Root dir of RM models containing metrics.json",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Where to save the png"
    )
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Directory {args.results_dir} not found.")
        exit(1)

    # 1. Collect Data
    df = collect_data(args.results_dir)

    if df.empty:
        print("No valid metrics found.")
        exit(0)

    df = df.sort_values(by="Size")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n--- Plotting Tasks ---")

    # 2. Plot Mean Score
    plot_single_task(df, "Mean", args.output_dir, RM_MEAN_BASELINES)

    # 3. Plot Individual Tasks
    for task_name in RM_TASK_BASELINES.keys():
        baselines = RM_TASK_BASELINES.get(task_name, {})
        plot_single_task(df, task_name, args.output_dir, baselines)

    # 4. Plot Combined View
    plot_combined_tasks(df, args.output_dir)
