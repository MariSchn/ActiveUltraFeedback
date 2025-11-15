import os
import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Directory containing your metric JSON files
results_dir = (
    "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/results/reward_models"
)

# Metrics to plot
metrics = ["Factuality", "Focus", "Math", "Precise IF", "Safety", "Ties", "score"]

# Collect metrics per step
metrics_by_step = defaultdict(lambda: defaultdict(list))

# Regex to extract step number from filename
step_pattern = re.compile(r"allenai_checkpoints_barna_(\d+)\.json")

for fname in os.listdir(results_dir):
    match = step_pattern.match(fname)
    if not match:
        continue
    step = int(match.group(1))
    with open(os.path.join(results_dir, fname), "r") as f:
        data = json.load(f)
        for metric in metrics:
            if metric in data:
                metrics_by_step[step][metric].append(data[metric])

# Bin steps into 250-step intervals, always include last step
all_steps = sorted(metrics_by_step.keys())
output_dir = "histograms"

for metric in metrics:
    steps = []
    values = []

    for step in all_steps:
        if metric in metrics_by_step[step]:
            # Some steps may have multiple values, take mean
            val = np.mean(metrics_by_step[step][metric])
            steps.append(step)
            values.append(val)

    if not steps:
        continue  # Skip metric if no data

    # Plot
    plt.figure(figsize=(16, 8))  # wider plot
    plt.plot(steps, values, marker="o", linestyle="-", label=metric)
    plt.xlabel("Training Step", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.title(f"{metric} over Steps scheduler=constant", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)

    # Add epoch markers
    max_step = max(all_steps)
    mid_step = max_step // 2
    y_top = max(values)

    plt.axvline(mid_step, color="red", linestyle="--", alpha=0.8)
    plt.text(
        mid_step - max_step * 0.005,
        y_top,
        "Epoch 1",
        color="red",
        fontsize=12,
        ha="right",
        va="bottom",
        rotation=0,
    )

    plt.axvline(max_step, color="blue", linestyle="--", alpha=0.8)
    plt.text(
        max_step - max_step * 0.005,
        y_top,
        "Epoch 2",
        color="blue",
        fontsize=12,
        ha="right",
        va="bottom",
        rotation=0,
    )

    # Save to file
    out_path = os.path.join(output_dir, f"{metric.replace(' ', '_')}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

print(f"Plots saved in {output_dir}")
