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

# 1. MAPPING RAW NAMES TO NICE NAMES
# Adjust the keys (left) to match exactly what is in your metrics.json files
TASK_NAME_MAP = {
    "gsm8k::tulu": "GSM8K",
    "ifeval::tulu": "IF Eval",
    "minerva_math::tulu": "Minerva Math",
    "truthfulqa::tulu": "Truthful QA",
    # Fallbacks if names are simple
    "gsm8k": "GSM8K",
    "ifeval": "IF Eval",
    "minerva_math": "Minerva Math",
    "truthfulqa": "Truthful QA",
}

# 2. BASELINES PER TASK
# Calculated from your provided SFT + Deltas
TASK_BASELINES = {
    "GSM8K": {
        "SFT": 0.758,
        "Random": 0.806,
        "UltraFeedback": 0.803,
        "MaxMin": 0.783,
        "DeltaQwen": 0.822,
    },
    "IF Eval": {
        "SFT": 0.713,
        "Random": 0.702,
        "UltraFeedback": 0.630,
        "MaxMin": 0.695,
        "DeltaQwen": 0.754,
    },
    "Minerva Math": {
        "SFT": 0.309,
        "Random": 0.342,
        "UltraFeedback": 0.348,
        "MaxMin": 0.398,
        "DeltaQwen": 0.377,
    },
    "Truthful QA": {
        "SFT": 0.468,
        "Random": 0.521,
        "UltraFeedback": 0.514,
        "MaxMin": 0.620,
        "DeltaQwen": 0.594,
    },
}

# 3. BASELINES FOR MEAN PLOT
# (SFT=0.562, Random=+0.032, UF=+0.012, MaxMin=+0.062, DeltaQwen=+0.074)
MEAN_BASELINES = {
    "SFT": 0.562,
    "Random": 0.594,
    "UltraFeedback": 0.574,
    "MaxMin": 0.624,
    "DeltaQwen": 0.636,
}

# ==============================================================================
#                                   HELPERS
# ==============================================================================


def extract_size_from_path(file_path):
    path_parts = os.path.normpath(file_path).split(os.sep)
    for part in reversed(path_parts):
        match = re.search(r"_(\d+)$", part)
        if match:
            return int(match.group(1))
    return None


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
                raw_name = parts[0].strip()
                score = float(parts[1].strip())

                # Clean up name using map
                clean_name = TASK_NAME_MAP.get(raw_name, raw_name)
                return clean_name, score
    except Exception as e:
        print(f"[Warn] Error parsing {file_path}: {e}")
        return None, None
    return None, None


def collect_data(root_dir):
    records = []
    search_pattern = os.path.join(root_dir, "**", "metrics.json")
    files = glob.glob(search_pattern, recursive=True)

    print(f"Found {len(files)} metrics files. Parsing...")

    for file_path in files:
        size = extract_size_from_path(file_path)
        if size is None:
            continue

        task, score = parse_metrics_file(file_path)
        if task is not None and score is not None:
            records.append({"Size": size, "Task": task, "Score": score})

    return pd.DataFrame(records)


def set_linear_xaxis(df):
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
        # Draw line
        plt_obj.axhline(y=val, color=colors[i], linestyle="--", alpha=0.7)
        # Add label on top of line at the far right
        plt_obj.text(
            x=x_pos_text,
            y=val,
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


def plot_single_task(df, task_name, output_dir):
    """Plots a single task with its specific baselines."""
    subset = df[df["Task"] == task_name].copy().sort_values(by="Size")

    if subset.empty:
        print(f"Skipping plot for {task_name} (No data)")
        return

    # Check for specific baselines
    baselines = TASK_BASELINES.get(task_name, {})

    # Connect to 0 (SFT)
    if "SFT" in baselines:
        sft_score = baselines["SFT"]
        start_point = pd.DataFrame([{"Size": 0, "Task": task_name, "Score": sft_score}])
        subset = pd.concat([start_point, subset], ignore_index=True)

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=subset, x="Size", y="Score", marker="o", linewidth=2.5, label=task_name
    )

    set_linear_xaxis(subset)

    # Draw Baselines
    draw_baselines(plt, baselines, subset["Size"].max())

    plt.title(f"{task_name} Score vs. Dataset Size", fontsize=16)
    plt.xlabel("Number of Training Samples", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()

    # Safe filename
    safe_name = task_name.replace(" ", "_").lower()
    save_path = os.path.join(output_dir, f"dpo_score_{safe_name}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved {task_name} plot to: {save_path}")
    plt.close()


def plot_combined_tasks(df, output_dir):
    """Plots all tasks on one chart (no baselines to avoid clutter)."""
    if df.empty:
        return

    plt.figure(figsize=(14, 8))

    sns.lineplot(data=df, x="Size", y="Score", hue="Task", marker="o", linewidth=2.5)

    set_linear_xaxis(df)

    plt.title("All Tasks: Scores vs. Dataset Size", fontsize=16)
    plt.xlabel("Number of Training Samples", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "dpo_all_tasks_combined.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved combined plot to: {save_path}")
    plt.close()


def plot_mean_score(df, output_dir):
    """Plots the Mean Score across all tasks."""
    if df.empty:
        return

    # Calculate Mean per Size
    mean_df = df.groupby("Size")["Score"].mean().reset_index()

    # Connect to 0 (SFT Mean)
    if "SFT" in MEAN_BASELINES:
        sft_score = MEAN_BASELINES["SFT"]
        start_point = pd.DataFrame([{"Size": 0, "Score": sft_score}])
        mean_df = pd.concat([start_point, mean_df], ignore_index=True)

    plt.figure(figsize=(12, 7))

    sns.lineplot(
        data=mean_df,
        x="Size",
        y="Score",
        marker="s",
        color="b",
        linewidth=3,
        label="Mean Task Score",
    )

    set_linear_xaxis(mean_df)

    # Draw Baselines
    draw_baselines(plt, MEAN_BASELINES, mean_df["Size"].max())

    plt.title("Mean Score vs. Dataset Size", fontsize=16)
    plt.xlabel("Number of Training Samples", fontsize=14)
    plt.ylabel("Mean Score", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "dpo_mean_score.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved mean score plot to: {save_path}")
    plt.close()


# ==============================================================================
#                                   MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
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

    print("\n--- Data Summary (Mean per Size) ---")
    print(df.groupby("Size")["Score"].mean())
    print("------------------------------------\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Plot Combined
    plot_combined_tasks(df, args.output_dir)

    # 3. Plot Mean
    plot_mean_score(df, args.output_dir)

    # 4. Plot Individual Tasks
    unique_tasks = df["Task"].unique()
    for task in unique_tasks:
        plot_single_task(df, task, args.output_dir)
