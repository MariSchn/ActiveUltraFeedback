import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

MODEL_TO_TEXT_COLOR = {
    "Qwen/Qwen2.5-0.5B-Instruct": "purple",
    "Qwen/Qwen2.5-72B-Instruct": "purple",
    "Qwen/Qwen3-0.6B": "purple",
    "Qwen/Qwen3-1.7B": "purple",
    "Qwen/Qwen3-14B": "purple",
    "Qwen/Qwen3-30B-A3B": "purple",
    "Qwen/Qwen3-32B": "purple",
    "Qwen/Qwen3-235B-A22B": "purple",
    "meta-llama/Llama-3.1-8B-Instruct": "blue",
    "meta-llama/Llama-3.2-1B-Instruct": "blue",
    "meta-llama/Llama-3.2-3B-Instruct": "blue",
    "meta-llama/Llama-3.3-70B-Instruct": "blue",
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1": "green",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": "green",
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1": "green",
    "allenai/Llama-3.1-Tulu-3-70B": "pink",
    "allenai/Llama-3.1-Tulu-3-405B": "pink",
    "allenai/OLMo-2-0325-32B-Instruct": "pink",
    "microsoft/Phi-4-mini-instruct": "green",
    "microsoft/phi-4": "green",
    "mistralai/Mistral-Small-24B-Instruct-2501": "orange",
    "mistralai/Mistral-Large-Instruct-2411": "orange",
    "google/gemma-3-1b-it": "goldenrod",
    "google/gemma-3-4b-it": "goldenrod",
    "google/gemma-3-12b-it": "goldenrod",
    "google/gemma-3-27b-it": "goldenrod",
    "CohereLabs/c4ai-command-a-03-2025": "black",
    "deepseek-ai/DeepSeek-V3": "blue",
    "moonshotai/Moonlight-16B-A3B-Instruct": "black",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "goldenrod",
}

# Plot constants
PLOT_SIZE = (16, 6)
SPACING = 1.25
BAR_WIDTH = 0.5

# Font constants
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 10

# Style constants
RED_COLOR = "red"
GREEN_COLOR = "green"
LIGHTBLUE_COLOR = "lightblue"
GRAY_COLOR = "gray"
DARKGREEN_COLOR = "darkgreen"
DARKRED_COLOR = "darkred"
BLACK_COLOR = "black"
DEFAULT_TEXT_COLOR = "black"

ALPHA_VALUE = 0.7
GRID_ALPHA = 0.3

# Boxplot constants
WHISKER_LINEWIDTH = 1.5
MEDIAN_LINEWIDTH = 2
BOXPLOT_SIDE_BY_SIDE_WIDTH = 0.5
BOXPLOT_WIDTH_MULTIPLIER = 0.85

# Output constants
DPI = 300


def has_columns(dataset: Dataset, columns: list[str]) -> bool:
    """Check if dataset has all specified columns."""
    return all(column in dataset.column_names for column in columns)


def get_x_positions(num_models: int) -> list[float]:
    """Generate x-axis positions for models with proper spacing."""
    return [i * SPACING for i in range(num_models)]


def apply_model_family_colors(ax, models: list[str]) -> None:
    """Apply color coding to x-axis labels based on model families."""
    for tick_label, model in zip(ax.get_xticklabels(), models):
        tick_label.set_color(MODEL_TO_TEXT_COLOR.get(model, DEFAULT_TEXT_COLOR))


def extract_preference_scores_per_model(
    dataset: Dataset, include_chosen: bool = True, include_rejected: bool = True
) -> tuple[dict[str, list], dict[str, list]]:
    """
    Extract chosen and rejected scores per model from a preference dataset.

    Returns:
        (model_to_chosen_scores, model_to_rejected_scores)
    """
    model_to_chosen_scores = {model: [] for model in MODEL_NAMES}
    model_to_rejected_scores = {model: [] for model in MODEL_NAMES}

    def extract_scores(sample):
        if include_chosen:
            model_to_chosen_scores[sample["chosen_model"]].append(
                sample["chosen_score"]
            )
        if include_rejected:
            model_to_rejected_scores[sample["rejected_model"]].append(
                sample["rejected_score"]
            )

    dataset.map(
        extract_scores,
        load_from_cache_file=False,
    )
    return model_to_chosen_scores, model_to_rejected_scores


def extract_annotated_scores_per_model(dataset: Dataset) -> dict[str, list]:
    """Extract scores per model from an annotated dataset with completions."""
    model_to_scores = {model: [] for model in MODEL_NAMES}

    def extract_scores(sample):
        for completion in sample["completions"]:
            model_to_scores[completion["model"]].append(completion["overall_score"])

    dataset.map(
        extract_scores,
        load_from_cache_file=False,
    )
    return model_to_scores


def style_boxplot_elements(
    bp: dict, face_color: str, whisker_color: str, median_color: str = BLACK_COLOR
) -> None:
    """Apply consistent styling to boxplot elements."""
    for patch in bp["boxes"]:
        patch.set_facecolor(face_color)
        patch.set_alpha(ALPHA_VALUE)

    for whisker in bp["whiskers"]:
        whisker.set(color=whisker_color, linewidth=WHISKER_LINEWIDTH)

    for cap in bp["caps"]:
        cap.set(color=whisker_color, linewidth=WHISKER_LINEWIDTH)

    for median in bp["medians"]:
        median.set(color=median_color, linewidth=MEDIAN_LINEWIDTH)


def plot_num_chosen_num_rejected_per_model(
    dataset: Dataset, output_path: str | None = None
):
    """
    Plots the number of times each model was used as chosen vs rejected.
    This is meant to be run on a preference dataset and only works
    if the dataset has 'chosen_model' and 'rejected_model' columns.
    """
    if not has_columns(dataset, ["chosen_model", "rejected_model"]):
        raise ValueError(
            "Dataset must have 'chosen_model', 'rejected_model' columns. "
            "Make sure that the dataset is a preference dataset"
        )

    model_to_chosen_counts = {model: 0 for model in MODEL_NAMES}
    model_to_rejected_counts = {model: 0 for model in MODEL_NAMES}

    def extract_data(sample):
        model_to_chosen_counts[sample["chosen_model"]] += 1
        model_to_rejected_counts[sample["rejected_model"]] += 1

    dataset.map(
        extract_data,
        load_from_cache_file=False,
    )

    models = MODEL_NAMES
    chosen_counts = [model_to_chosen_counts[model] for model in models]
    rejected_counts = [model_to_rejected_counts[model] for model in models]

    # Create the plot
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    x_positions = get_x_positions(len(models))

    ax.bar(
        [x - BAR_WIDTH / 2 for x in x_positions],
        chosen_counts,
        BAR_WIDTH,
        label="Chosen",
        color=GREEN_COLOR,
    )
    ax.bar(
        [x + BAR_WIDTH / 2 for x in x_positions],
        rejected_counts,
        BAR_WIDTH,
        label="Rejected",
        color=RED_COLOR,
    )

    ax.set_xlabel("Model", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Count", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_title(
        "Chosen vs Rejected Counts per Model",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=45, ha="right")

    apply_model_family_colors(ax, models)

    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.grid(axis="y", alpha=GRID_ALPHA, linestyle="--")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Num Chosen Num Rejected Per Model Plot saved to {output_path}")

    return fig


def plot_score_boxplot_per_model(
    dataset: Dataset,
    output_path: str | None = None,
    plot_chosen: bool = True,
    plot_rejected: bool = True,
):
    """
    Creates a boxplot for the scores of each model.
    This can run on both fully annotated datasets and preference datasets.
    This means it needs to either have:
    - a "completions" column which has an attribute "overall_score"
    - a "chosen_model", "rejected_model", "chosen_score", "rejected_score" columns
    """
    # Extract scores based on dataset type
    if has_columns(dataset, ["completions"]):
        model_to_scores = extract_annotated_scores_per_model(dataset)
    elif has_columns(
        dataset, ["chosen_model", "rejected_model", "chosen_score", "rejected_score"]
    ):
        model_to_chosen, model_to_rejected = extract_preference_scores_per_model(
            dataset, include_chosen=plot_chosen, include_rejected=plot_rejected
        )
        # Combine chosen and rejected scores
        model_to_scores = {model: [] for model in MODEL_NAMES}
        for model in MODEL_NAMES:
            model_to_scores[model].extend(model_to_chosen[model])
            model_to_scores[model].extend(model_to_rejected[model])
    else:
        raise ValueError(
            "Dataset must have 'completions' or 'chosen_model', 'rejected_model', "
            "'chosen_score', 'rejected_score' columns. "
            "Make sure that the dataset is a fully annotated dataset or a preference dataset"
        )

    models = MODEL_NAMES
    scores_data = [model_to_scores[model] for model in models]

    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    x_positions = get_x_positions(len(models))
    bp = ax.boxplot(
        scores_data,
        positions=x_positions,
        tick_labels=models,
        patch_artist=True,
        showfliers=False,
    )

    # Customize boxplot
    style_boxplot_elements(bp, LIGHTBLUE_COLOR, GRAY_COLOR, RED_COLOR)

    # Set title based on what's being plotted
    if plot_chosen and plot_rejected:
        title = "Score Distribution per Model"
    elif plot_chosen:
        title = "Chosen Score Distribution per Model"
    elif plot_rejected:
        title = "Rejected Score Distribution per Model"

    ax.set_xlabel("Model", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Overall Score", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=45, ha="right")

    apply_model_family_colors(ax, models)

    ax.grid(axis="y", alpha=GRID_ALPHA, linestyle="--")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Score Boxplot saved to {output_path}")

    return fig


def plot_score_boxplot_chosen_vs_rejected_per_model(
    dataset: Dataset, output_path: str | None = None
):
    """
    Creates side-by-side boxplots comparing chosen vs rejected scores for each model.
    This only works on preference datasets with 'chosen_model', 'rejected_model',
    'chosen_score', 'rejected_score' columns.
    """
    if not has_columns(
        dataset, ["chosen_model", "rejected_model", "chosen_score", "rejected_score"]
    ):
        raise ValueError(
            "Dataset must have 'chosen_model', 'rejected_model', 'chosen_score', "
            "'rejected_score' columns. Make sure that the dataset is a preference dataset"
        )

    # Extract scores
    model_to_chosen, model_to_rejected = extract_preference_scores_per_model(
        dataset, include_chosen=True, include_rejected=True
    )

    models = MODEL_NAMES
    chosen_scores_data = [model_to_chosen[model] for model in models]
    rejected_scores_data = [model_to_rejected[model] for model in models]

    fig, ax = plt.subplots(figsize=PLOT_SIZE)

    # Create positions for side-by-side boxplots
    x_positions = get_x_positions(len(models))
    offset = BOXPLOT_SIDE_BY_SIDE_WIDTH / 2
    box_width = BOXPLOT_SIDE_BY_SIDE_WIDTH * BOXPLOT_WIDTH_MULTIPLIER

    # Create chosen boxplots
    bp_chosen = ax.boxplot(
        chosen_scores_data,
        positions=[x - offset for x in x_positions],
        widths=box_width,
        patch_artist=True,
        showfliers=False,
    )

    # Create rejected boxplots
    bp_rejected = ax.boxplot(
        rejected_scores_data,
        positions=[x + offset for x in x_positions],
        widths=box_width,
        patch_artist=True,
        showfliers=False,
    )

    # Style boxplots
    style_boxplot_elements(bp_chosen, GREEN_COLOR, DARKGREEN_COLOR, BLACK_COLOR)
    style_boxplot_elements(bp_rejected, RED_COLOR, DARKRED_COLOR, BLACK_COLOR)

    # Customize Plot
    ax.set_xlabel("Model", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Overall Score", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_title(
        "Chosen vs Rejected Score Distribution per Model",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=45, ha="right")

    apply_model_family_colors(ax, models)

    # Create legend
    legend_elements = [
        Patch(facecolor=GREEN_COLOR, alpha=ALPHA_VALUE, label="Chosen"),
        Patch(facecolor=RED_COLOR, alpha=ALPHA_VALUE, label="Rejected"),
    ]
    ax.legend(handles=legend_elements, fontsize=LEGEND_FONTSIZE)

    ax.grid(axis="y", alpha=GRID_ALPHA, linestyle="--")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Chosen vs Rejected Score Boxplot saved to {output_path}")

    return fig


def calculate_statistics(dataset: Dataset, output_path: str | None = None) -> dict:
    """
    Calculates statistics for the dataset. Currently calculates:

    - Mean Score Rejected
    - Mean Score Chosen
    - Mean Score Delta (Chosen - Rejected)

    This only works on preference datasets.
    """
    if not has_columns(
        dataset, ["chosen_model", "rejected_model", "chosen_score", "rejected_score"]
    ):
        raise ValueError(
            "Dataset must have 'chosen_model', 'rejected_model', 'chosen_score', "
            "'rejected_score' columns. Make sure that the dataset is a preference dataset"
        )

    chosen_scores = []
    rejected_scores = []

    def extract_scores(sample):
        chosen_scores.append(sample["chosen_score"])
        rejected_scores.append(sample["rejected_score"])

    dataset.map(extract_scores, load_from_cache_file=False)

    statistics = {
        "mean_score_chosen": float(np.mean(chosen_scores)),
        "mean_score_rejected": float(np.mean(rejected_scores)),
        "mean_score_delta": float(np.mean(chosen_scores) - np.mean(rejected_scores)),
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(statistics, f, indent=2)
            print(f"Statistics saved to {output_path}")

    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = os.path.join(args.dataset_path, "plots")
        os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_from_disk(args.dataset_path)
    is_preference = "chosen" in dataset.column_names

    if is_preference:
        # Preference Dataset Plots
        plot_num_chosen_num_rejected_per_model(
            dataset, os.path.join(args.output_dir, "chosen_rejected_counts.png")
        )
        plot_score_boxplot_per_model(
            dataset,
            os.path.join(args.output_dir, "score_boxplot.png"),
            plot_chosen=True,
            plot_rejected=True,
        )
        plot_score_boxplot_chosen_vs_rejected_per_model(
            dataset,
            os.path.join(args.output_dir, "score_boxplot_chosen_vs_rejected.png"),
        )
        calculate_statistics(dataset, os.path.join(args.output_dir, "statistics.json"))
    else:
        # Annotated Dataset Plots
        plot_score_boxplot_per_model(
            dataset, os.path.join(args.output_dir, "score_boxplot_annotated.png")
        )
