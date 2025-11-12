import os
import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# List your results directories here
results_dirs = [
    # "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/results/reward_models/constant",
    # "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/results/reward_models/cosine",
    # "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/results/preference_random_llama_checkpoints_good",
    "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/results/reward_models/",
    # "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/results/reward_models/skywork_20000.json",
    # "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/results/reward_models/skywork_30000.json",
    # "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/results/reward_models/skywork_50000.json",
    # "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/results/reward_models/skywork_60000.json",
    # "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/results/reward_models/skywork_70000.json",
    # "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/results/reward_models/skywork_78000.json",
]

metrics = ["Factuality", "Focus", "Math",
           "Precise IF", "Safety", "Ties", "score"]
step_pattern = re.compile(r"skywork_(\d+)\.json")
output_dir = "multi_histograms"
os.makedirs(output_dir, exist_ok=True)

# Collect metrics for each directory
metrics_by_dir = {}
all_steps_set = set()
for dir_path in results_dirs:
    metrics_by_step = defaultdict(lambda: defaultdict(list))
    for fname in os.listdir(dir_path):
        match = step_pattern.match(fname)
        if not match:
            continue
        step = int(match.group(1))
        all_steps_set.add(step)
        with open(os.path.join(dir_path, fname), "r") as f:
            data = json.load(f)
            for metric in metrics:
                if metric in data:
                    metrics_by_step[step][metric].append(data[metric])
    metrics_by_dir[dir_path] = metrics_by_step

all_steps = sorted(all_steps_set)

# Plot each metric, overlaying curves from all directories
for metric in metrics:
    plt.figure(figsize=(16, 8))
    for dir_path in results_dirs:
        metrics_by_step = metrics_by_dir[dir_path]
        steps = []
        values = []
        for step in sorted(metrics_by_step.keys()):
            if metric in metrics_by_step[step]:
                val = np.mean(metrics_by_step[step][metric])
                steps.append(step)
                values.append(val)
        if steps:
            label = os.path.basename(dir_path)
            plt.plot(steps, values, marker="o", linestyle="-", label=label)
    plt.xlabel("Subset size", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.title(f"{metric} over Steps (multiple runs)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)

    # Add epoch markers (red vertical lines)
    if all_steps:
        max_step = max(all_steps)
        mid_step = max_step // 2
        y_bottom = plt.gca().get_ylim()[0]
        # small offset above bottom
        y_offset = (plt.gca().get_ylim()[1] - y_bottom) * 0.02

        # plt.axvline(mid_step, color="red", linestyle="--", alpha=0.8)
        # plt.text(mid_step, y_bottom + y_offset, "Epoch 1", color="red", fontsize=12,
        #          ha="center", va="bottom", rotation=0)
        # plt.axvline(max_step, color="red", linestyle="--", alpha=0.8)
        # plt.text(max_step, y_bottom + y_offset, "Epoch 2", color="red", fontsize=12,
        #          ha="center", va="bottom", rotation=0)

    out_path = os.path.join(
        output_dir, f"{metric.replace(' ', '_')}_multi.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

print(f"Overlaid plots saved in {output_dir}")
