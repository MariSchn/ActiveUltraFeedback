import argparse
import os
import matplotlib.pyplot as plt
from datasets import load_from_disk, Dataset

MODEL_NAMES = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-235B-A22B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
    "allenai/Llama-3.1-Tulu-3-70B",
    "allenai/Llama-3.1-Tulu-3-405B",
    "allenai/OLMo-2-0325-32B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "microsoft/phi-4",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistralai/Mistral-Large-Instruct-2411",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "CohereLabs/c4ai-command-a-03-2025",
    "deepseek-ai/DeepSeek-V3",
    "moonshotai/Moonlight-16B-A3B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]

DOUBLE_BAR_PLOT_BAR_WIDTH = 0.35
RED_COLOR = "red"
GREEN_COLOR = "green"


def has_columns(dataset: Dataset, columns: list[str]) -> bool:
    return all(column in dataset.column_names for column in columns)


def plot_num_chosen_num_rejected_per_model(
    dataset: Dataset, output_path: str = "./plots/chosen_rejected_counts.png"
):
    """
    Plots the number of times each model was used as chosen vs rejected.
    Only works if `dataset` is a preference dataset with 'chosen_model' and 'rejected_model' columns.
    """
    if not has_columns(dataset, ["chosen_model", "rejected_model"]):
        raise ValueError(
            "Dataset must have 'chosen_model', 'rejected_model' columns."
            "Make sure that the dataset is a preference dataset"
        )

    model_to_chosen_counts = {model: 0 for model in MODEL_NAMES}
    model_to_rejected_counts = {model: 0 for model in MODEL_NAMES}

    def extract_data(sample):
        model_to_chosen_counts[sample["chosen_model"]] += 1
        model_to_rejected_counts[sample["rejected_model"]] += 1

    dataset.map(extract_data)

    models = MODEL_NAMES
    chosen_counts = [model_to_chosen_counts[model] for model in models]
    rejected_counts = [model_to_rejected_counts[model] for model in models]

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    x_positions = range(len(models))

    ax.bar(
        [x - DOUBLE_BAR_PLOT_BAR_WIDTH / 2 for x in x_positions],
        chosen_counts,
        DOUBLE_BAR_PLOT_BAR_WIDTH,
        label="Chosen",
        color=GREEN_COLOR,
    )
    ax.bar(
        [x + DOUBLE_BAR_PLOT_BAR_WIDTH / 2 for x in x_positions],
        rejected_counts,
        DOUBLE_BAR_PLOT_BAR_WIDTH,
        label="Rejected",
        color=RED_COLOR,
    )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title("Chosen vs Rejected Counts per Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./plots")
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)
    plot_num_chosen_num_rejected_per_model(
        dataset, os.path.join(args.output_dir, "chosen_rejected_counts.png")
    )
