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

# 1. BASELINES FOR MEAN SCORE
RM_MEAN_BASELINES = {
    "SFT": 0.290,  # Retained from previous context to anchor x=0
    "DeltaQwen": 0.390,
    "MaxMin": 0.608,
    "Random": 0.568,
    "UltraFeedback": 0.577,
}

# 2. BASELINES FOR INDIVIDUAL TASKS
RM_TASK_BASELINES = {
    "Factuality": {
        "SFT": 0.316,
        "DeltaQwen": 0.511,
        "MaxMin": 0.693,
        "Random": 0.759,
        "UltraFeedback": 0.759,
    },
    "Focus": {
        "SFT": 0.277,
        "DeltaQwen": 0.243,
        "MaxMin": 0.760,
        "Random": 0.486,
        "UltraFeedback": 0.465,
    },
    "Math": {
        "SFT": 0.445,
        "DeltaQwen": 0.473,
        "MaxMin": 0.601,
        "Random": 0.601,
        "UltraFeedback": 0.658,
    },
    "Precise IF": {
        "SFT": 0.261,
        "DeltaQwen": 0.328,
        "MaxMin": 0.384,
        "Random": 0.394,
        "UltraFeedback": 0.375,
    },
    "Safety": {
        "SFT": 0.347,
        "DeltaQwen": 0.563,
        "MaxMin": 0.717,
        "Random": 0.764,
        "UltraFeedback": 0.828,
    },
    "Ties": {
        "SFT": 0.095,
        "DeltaQwen": 0.221,
        "MaxMin": 0.495,
        "Random": 0.405,
        "UltraFeedback": 0.379,
    },
}

# ==============================================================================
#                                   HELPERS
# ==============================================================================


def extract_info_from_path(file_path):
    """
    Walks up the path directories to find a folder matching:
    '1191784-delta_qwen_10000' (ID-Method_Size)
    """
    path_parts = os.path.normpath(file_path).split(os.sep)

    # Iterate reversed (from file up to root)
    for part in reversed(path_parts):
        # Regex to match: Digits + hyphen + Method + underscore + Digits
        match = re.match(r"^\d+-(.+)_(\d+)$", part)
        if match:
            method = match.group(1)  # e.g., delta_qwen
            size = int(match.group(2))  # e.g., 10000
            return method, size

        # Fallback for folder names without ID (e.g. random_60829)
        match_simple = re.match(r"^(.+)_(\d+)$", part)
        if match_simple and not match_simple.group(1).isdigit():
            return match_simple.group(1), int(match_simple.group(2))

    return None, None


def parse_metrics_file(file_path):
    """
    Parses metrics.json and returns a dictionary of all scores.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data  # Returns dict like {"Factuality": 0.7, "Mean": 0.5...}
    except Exception as e:
        return None


def collect_data_from_single_dir(root_dir):
    records = []

    if not os.path.exists(root_dir):
        print(f"Warning: Directory {root_dir} does not exist. Skipping.")
        return pd.DataFrame()

    search_pattern = os.path.join(root_dir, "**", "metrics.json")
    files = glob.glob(search_pattern, recursive=True)

    print(f"Scanning {root_dir}... Found {len(files)} metrics files.")

    for file_path in files:
        # 1. Extract Method and Size from path
        method, size = extract_info_from_path(file_path)

        if size is None:
            continue

        # 2. Extract Scores from JSON
        data = parse_metrics_file(file_path)
        if data:
            # Iterate over all keys in the json (Factuality, Mean, etc.)
            for task_name, score in data.items():
                if isinstance(score, (int, float)):
                    records.append(
                        {
                            "Method": method,
                            "Size": size,
                            "Task": task_name,
                            "Score": float(score),
                            "Source": root_dir,
                        }
                    )

    return pd.DataFrame(records)


def set_linear_xaxis(df, plt_obj):
    """Sets x-axis to linear scale with ticks every 5000."""
    max_val = df["Size"].max()
    upper_bound = int(max_val) + 5000
    ticks = np.arange(0, upper_bound, 5000)
    plt_obj.xticks(ticks)
    plt_obj.xlim(0, upper_bound)


def draw_baselines(plt_obj, baselines, x_pos_text):
    """Helper to draw horizontal lines and labels."""
    colors = sns.color_palette("tab10", len(baselines))
    for i, (name, val) in enumerate(baselines.items()):
        plt_obj.axhline(y=val, color=colors[i], linestyle="--", alpha=0.6, linewidth=1)
        plt_obj.text(
            x=x_pos_text,
            y=val,
            s=f"{name}",
            color=colors[i],
            fontweight="bold",
            fontsize=8,
            ha="right",
            va="bottom",
        )


# ==============================================================================
#                                 PLOTTING
# ==============================================================================


def plot_task_comparison(
    df, task_name, output_dir, baselines, color_map, show_baselines=True
):
    """
    Plots a single task (or Mean) comparing multiple Methods.
    """
    # Filter data for this task
    subset = df[df["Task"] == task_name].copy().sort_values(by="Size")

    if subset.empty:
        # print(f"Skipping plot for {task_name} (No data)")
        return

    # --- CONNECT TO 0 (SFT) FOR ALL METHODS ---
    if baselines and "SFT" in baselines:
        sft_score = baselines["SFT"]
        unique_methods = subset["Method"].unique()
        start_points = []
        for method in unique_methods:
            start_points.append(
                {"Size": 0, "Task": task_name, "Score": sft_score, "Method": method}
            )
        subset = pd.concat([pd.DataFrame(start_points), subset], ignore_index=True)

    plt.figure(figsize=(10, 6))

    # Plot with Hue=Method
    sns.lineplot(
        data=subset,
        x="Size",
        y="Score",
        hue="Method",
        marker="o",
        linewidth=2.5,
        palette=color_map,  # Consistent colors
    )

    set_linear_xaxis(subset, plt)

    if show_baselines and baselines:
        draw_baselines(plt, baselines, subset["Size"].max())

    plt.title(f"RM {task_name}: Method Comparison", fontsize=16)
    plt.xlabel("Number of Training Samples", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(title="Strategy")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()

    # Safe filename
    safe_name = task_name.replace(" ", "_").lower()
    save_path = os.path.join(output_dir, f"rm_compare_{safe_name}.png")
    plt.savefig(save_path, dpi=300)
    # print(f"Saved {task_name} plot to: {save_path}")
    plt.close()


# ==============================================================================
#                                   MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RM evaluation results.")
    # Updated to accept multiple directories
    parser.add_argument(
        "--results_dirs",
        nargs="+",
        required=True,
        help="List of root directories containing RM models",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Where to save the png"
    )
    parser.add_argument(
        "--hide_baselines", action="store_true", help="Hide horizontal baseline lines"
    )
    args = parser.parse_args()

    all_data_frames = []

    print(f"Aggregating data from {len(args.results_dirs)} directories...")

    # 1. Collect Data from all dirs
    for result_dir in args.results_dirs:
        df_part = collect_data_from_single_dir(result_dir)
        if not df_part.empty:
            all_data_frames.append(df_part)

    if not all_data_frames:
        print("No valid metrics found in any directory.")
        exit(0)

    # Merge
    df = pd.concat(all_data_frames, ignore_index=True)
    df = df.sort_values(by="Size")

    # --- CREATE CONSISTENT COLOR MAP ---
    unique_methods = sorted(df["Method"].unique())
    # Create a palette with enough distinct colors
    palette = sns.color_palette("bright", n_colors=len(unique_methods))
    # Map each method name to a specific color
    method_color_map = dict(zip(unique_methods, palette))

    print("\n--- Data Summary ---")
    print(f"Sources: {[d for d in args.results_dirs]}")
    print(f"Methods: {unique_methods}")
    print(f"Tasks:   {df['Task'].unique()}")
    print("--------------------\n")

    os.makedirs(args.output_dir, exist_ok=True)
    show_baselines = not args.hide_baselines

    # 2. Plot Mean Score (if "Mean" exists in your json keys)
    if "Mean" in df["Task"].unique():
        plot_task_comparison(
            df,
            "Mean",
            args.output_dir,
            RM_MEAN_BASELINES,
            method_color_map,
            show_baselines,
        )

    # 3. Plot Individual Tasks
    for task_name in RM_TASK_BASELINES.keys():
        baselines = RM_TASK_BASELINES.get(task_name, {})
        plot_task_comparison(
            df, task_name, args.output_dir, baselines, method_color_map, show_baselines
        )

    print(f"Done. Plots saved to {os.path.abspath(args.output_dir)}")
