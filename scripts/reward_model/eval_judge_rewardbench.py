"""
This script is used to annotate the completions generated from the generate_completions.py script.
It uses a LLM as a judge to rate the completions based on the aspects defined in the configs.py file and provides critique/feedback for each completion.

Adapted from https://github.com/allenai/reward-bench/blob/main/scripts/run_v2.py

Example run command:
python -m scripts.eval_judge_rewardbench \
    --model_path="Qwen/Qwen3-32B" \
    --output_path /iopsstor/scratch/cscs/smarian/data/judge_rewardbench/Qwen3-32B \
    --max_tokens 24000 \
    --model_class vllm \
    --temperature 0.0 \
    --top_p 0.1

python -m scripts.eval_judge_rewardbench \
    --model_path="meta-llama/Llama-3.3-70B-Instruct" \
    --output_path /iopsstor/scratch/cscs/smarian/data/judge_rewardbench/Llama-3.3-70B-Instruct \
    --max_tokens 24000 \
    --model_class vllm \
    --temperature 0.0 \
    --top_p 0.1
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

from vllm import SamplingParams, LLM
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from activeuf.utils import set_seed, get_logger, setup, get_response_texts, load_model
from activeuf.oracle.prompts import (
    PREFERENCE_ANNOTATION_SYSTEM_PROMPT,
    INSTRUCTION_FOLLOWING_ANNOTATION_SYSTEM_PROMPT,
    HONESTY_ANNOTATION_SYSTEM_PROMPT,
    TRUTHFULNESS_ANNOTATION_SYSTEM_PROMPT,
    HELPFULNESS_ANNOTATION_SYSTEM_PROMPT,
)

ASPECT2ANNOTATION_PROMPT = {
    "instruction_following": INSTRUCTION_FOLLOWING_ANNOTATION_SYSTEM_PROMPT,
    "honesty": HONESTY_ANNOTATION_SYSTEM_PROMPT,
    "truthfulness": TRUTHFULNESS_ANNOTATION_SYSTEM_PROMPT,
    "helpfulness": HELPFULNESS_ANNOTATION_SYSTEM_PROMPT,
}

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    # fmt: off

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="Where to export the results and annotated dataset")
    parser.add_argument("--model_path", type=str, required=True, help="The (Huggingface) to the model to use for annotation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random sampling")

    parser.add_argument("--model_class", type=str, default="vllm", help="The class which is used to perform inference (e.g. transformers, pipeline, vllm)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="The maximum number of tokens for LLM responses")
    parser.add_argument("--max_num_gpus", type=int, default=4, help="The maximum number of GPUs to use")
    parser.add_argument("--num_nodes", type=int, default=1, help="The number of nodes to use")
    parser.add_argument("--temperature", type=float, default=1.0, help="The temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="The top_p for sampling")

    parser.add_argument("--debug", action="store_true", help="If set, will only annotate the first few samples")

    args = parser.parse_args()

    return args


def calculate_probabilities(raw_output, tokenizer, target_words):
    target_token_ids = [
        tokenizer.encode(t, add_special_tokens=False)[0] for t in target_words
    ]

    word_probabilities = []

    for output in raw_output:
        logprobs = output.outputs[0].logprobs

        logprob_dict = logprobs[0]

        token_logprobs = {}
        for t, tid in zip(target_words, target_token_ids):
            token_logprobs[t] = logprob_dict.get(tid, -float("inf"))

        def get_logprob_value(lp):
            return lp.logprob if hasattr(lp, "logprob") else lp

        exp_values = [np.exp(get_logprob_value(lp)) for lp in token_logprobs.values()]

        total = sum(exp_values)
        prob_dict = {
            k: float(v) / total for k, v in zip(token_logprobs.keys(), exp_values)
        }

        word_probabilities.append(prob_dict)

    return word_probabilities


def calculate_probabilities_openai(raw_output, target_words):
    """
    Calculates the probabilities of target words from OpenAI API outputs.
    """
    word_probabilities = []

    for output in raw_output:
        first_token_logprobs = output.choices[0].logprobs.content[0].top_logprobs

        token_logprobs = {}
        for token_logprob in first_token_logprobs:
            token_logprobs[token_logprob.token] = token_logprob.logprob

        # Find logprobs for our target words
        target_logprobs = {}
        for word in target_words:
            logprob = token_logprobs.get(word, -float("inf"))
            target_logprobs[word] = logprob

        exp_values = [np.exp(lp) for lp in target_logprobs.values()]
        total = sum(exp_values)

        if total == 0:
            # Avoid division by zero if no target words were found
            prob_dict = {k: 0.0 for k in target_logprobs.keys()}
        else:
            prob_dict = {
                k: float(v) / total for k, v in zip(target_logprobs.keys(), exp_values)
            }

        word_probabilities.append(prob_dict)

    print(word_probabilities[0])

    return word_probabilities


def probabilities_to_score(probabilities: Dict[str, float]) -> float:
    expectation = 0
    for rating in range(1, 6):
        expectation += rating * probabilities.get(str(rating))
    return expectation


def grouped_probabilities_to_score(
    grouped_probabilities: List[Dict[str, float]],
) -> float:
    return sum(probabilities_to_score(prob) for prob in grouped_probabilities) / len(
        grouped_probabilities
    )


def reroll_and_score_dataset(
    dataset, total_completions, cols_to_combine=["text", "scores"]
):
    """
    Taken from rewardbench repo: https://github.com/allenai/reward-bench/blob/a0096928eb1d54edd3f6361374708cc68e738b79/rewardbench/utils.py#L535
    """
    # Convert to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()

    # Validate that sum of total_completions matches dataset length
    if sum(total_completions) != len(df):
        raise ValueError(
            f"Sum of total_completions ({sum(total_completions)}) does not equal dataset length ({len(df)})"
        )

    rerolled_rows = []
    current_idx = 0

    # Process each group with its specified number of completions
    for group_size in total_completions:
        group = df.iloc[current_idx : current_idx + group_size]

        # Create new row
        new_row = {}
        # print(group['scores'])
        # Handle text and score columns - combine into lists
        for col in cols_to_combine:
            new_row[col] = group[col].tolist()

        # penalty for ties
        scores = new_row["scores"]
        max_val = np.max(scores)
        new_row["results"] = (
            (1 / np.sum(scores == max_val)) if scores[0] == max_val else 0
        )

        # new_row["results"] = 1 if np.argmax(new_row["scores"]) == 0 else 0

        # Handle all other columns - verify they're identical and take first value
        other_columns = [col for col in df.columns if col not in cols_to_combine]
        for col in other_columns:
            values = group[col].unique()
            if len(values) != 1:
                raise ValueError(
                    f"Column {col} has different values within group at index {current_idx}: {values}"
                )
            new_row[col] = values[0]

        rerolled_rows.append(new_row)
        current_idx += group_size

    # Create new dataset
    rerolled_df = pd.DataFrame(rerolled_rows)
    rerolled_dataset = Dataset.from_pandas(rerolled_df)

    return rerolled_dataset


def check_tokenizer_chat_template(tokenizer):
    """
    Check if tokenizer has non none chat_template attribute.
    """
    if hasattr(tokenizer, "chat_template"):
        if tokenizer.chat_template is not None:
            return True
    return False


# Helper function for scoring ties subset
def _compute_prompt_stats(
    samples: List[Tuple[bool, float]],
) -> Tuple[bool, float | None, float | None]:
    """
    Given a list of (is_correct, score) tuples for one prompt,
    return:
        accurate ................ True if every correct answer outscores the best wrong one
        different_correct_margin  Spread between best and worst correct answers (None if <2)
        correct_incorrect_margin  Gap between worst correct and best wrong (None if N/A)
    """
    correct_scores = [s for is_corr, s in samples if is_corr]
    incorrect_scores = [s for is_corr, s in samples if not is_corr]
    best_correct = max(correct_scores)
    worst_correct = min(correct_scores)
    best_incorrect = max(incorrect_scores)

    # Calculate the margins with correct scores, and also the margin between correct and incorrect scores
    different_correct_margin = (
        best_correct - worst_correct if len(correct_scores) > 1 else None
    )
    correct_incorrect_margin = worst_correct - best_incorrect
    accurate = correct_incorrect_margin > 0

    return accurate, different_correct_margin, correct_incorrect_margin


def process_single_model(dataset):
    """
    Process a single-model ties evaluation dataset and return
        (dataset_with_results_column, overall_score)
    Each row in the dataset contains a list of "scores", where the first "num_correct" correspond to
        correct answers, and the rest are incorrect. The "id" field is formatted as "sample_type:prompt_id",
        where sample_type is either "ref" for reference prompts with 1 correct answer or "tied" for tied samples
        with multiple correct answers.
    Overall score is essentially 60% accuracy, 40% margin. Accuracy is broken down equally
        across ref and tied accuracy, while margin is broken down into whether the margin between
        correct answers < margin between correct and incorrect answers for tied prompts only (correctness_preferred)
        and whether this margin also holds when the margin between correct and incorrect answers is the min of the
        margin for a tied prompt and its associated reference prompt (correctness_preferred_hard).
    """
    grouped_samples: Dict[Tuple[str, int], List[Tuple[bool, float]]] = defaultdict(list)

    for sample in dataset:
        # Split samples into ref and tied
        sample_type, prompt_id_str = sample["id"].split(":")
        prompt_id = int(prompt_id_str)

        # Each score position i is “correct” if i < num_correct
        for i, raw_score in enumerate(sample["scores"]):
            score = raw_score[0] if isinstance(raw_score, list) else raw_score
            grouped_samples[(sample_type, prompt_id)].append(
                (i < sample["num_correct"], score)
            )

    # Calculate per-prompt stats
    ref_stats = {}
    tied_stats = {}

    for (sample_type, prompt_id), samples in grouped_samples.items():
        stats = _compute_prompt_stats(samples)
        if sample_type == "ref":
            ref_stats[prompt_id] = stats
        else:  # "tied"
            tied_stats[prompt_id] = stats

    # Calculate global metrics
    # Average accuracy (element 0 of each tuple) over ref and tied samples
    ref_accuracy = np.mean([s[0] for s in ref_stats.values()]) if ref_stats else 0.0
    tied_accuracy = np.mean([s[0] for s in tied_stats.values()]) if tied_stats else 0.0

    # Margins: compute whether margin within correct answers < margin between correct and incorrect answers
    all_prompts = set(ref_stats) & set(tied_stats)

    # correct margin is element 1 in stats tuple, correct-incorrect margin is element 2
    diff_corr_margin = np.array([tied_stats[pid][1] for pid in all_prompts])
    corr_incorrect_ties = np.array([tied_stats[pid][2] for pid in all_prompts])
    corr_incorrect_ref = np.array([ref_stats[pid][2] for pid in all_prompts])

    correctness_preferred = np.mean(corr_incorrect_ties > diff_corr_margin)
    correctness_preferred_hard = np.mean(
        np.minimum(corr_incorrect_ref, corr_incorrect_ties) > diff_corr_margin
    )

    # Tie-breaking term, optional, not much effect in practice
    # Normalised gap, then tanh to keep it in (‑1, 1)
    margin_scores = np.tanh(
        np.minimum(corr_incorrect_ref, corr_incorrect_ties) / diff_corr_margin - 1
    )
    # if nan (divide by 0), set to 0
    margin_scores = np.nan_to_num(margin_scores, nan=0.0)
    correctness_margin_score = float(np.mean(margin_scores))

    # Compute the overall score
    overall_score = (
        0.30 * tied_accuracy
        + 0.30 * ref_accuracy
        + 0.20 * correctness_preferred
        + 0.20 * correctness_preferred_hard
        + 0.01 * correctness_margin_score
    )

    # Package results — there is less of a sense of per-prompt results for the Ties subset,
    # as overall_score is computed across the subset, so set "results" to None for clarity
    if "results" in dataset.column_names:
        dataset = dataset.remove_columns(["results"])
    results_dataset = dataset.add_column("results", [None] * len(dataset))

    return results_dataset, float(overall_score)


def run(dataset: Dataset, args: argparse.Namespace):
    logger.info("Unrolling dataset for annotation")
    unrolled_dataset = []
    subsets = []
    total_completions = []
    num_correct = []
    ids = []
    text = []  # Is also tokenized in actual rewardbench code but not necessary here
    for i, sample in enumerate(dataset):
        total_completions.append(sample["total_completions"])
        num_correct.append(sample["num_correct"])
        for chosen_completion in sample["chosen"]:
            unrolled_dataset.append(
                {
                    "id": sample["id"],
                    "row": i,
                    "type": "chosen",
                    "prompt": sample["prompt"],
                    "completion": chosen_completion,
                }
            )
            subsets.append(sample["subset"])
            ids.append(sample["id"])
            text.append(sample["prompt"] + "\n" + chosen_completion)
        for rejected_completion in sample["rejected"]:
            unrolled_dataset.append(
                {
                    "id": sample["id"],
                    "row": i,
                    "type": "rejected",
                    "prompt": sample["prompt"],
                    "completion": rejected_completion,
                }
            )
            subsets.append(sample["subset"])
            ids.append(sample["id"])
            text.append(sample["prompt"] + "\n" + rejected_completion)

    logger.info("Creating messages")
    conversations = []
    for sample in unrolled_dataset:
        for _, annotation_prompt in ASPECT2ANNOTATION_PROMPT.items():
            conversations.append(
                [
                    {
                        "role": "system",
                        "content": PREFERENCE_ANNOTATION_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": annotation_prompt.format(
                            prompt=sample["prompt"],
                            completion=sample["completion"],
                        ),
                    },
                ]
            )

    logger.info(f"Loading {args.model_path} for annotation")
    model, tokenizer = load_model(
        args.model_path,
        args.model_class,
        max_num_gpus=args.max_num_gpus,
        num_nodes=args.num_nodes,
        data_parallel_size=1,
    )

    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )

    logger.info("Running inference for annotation")
    sampling_params = SamplingParams(
        max_tokens=64,
        temperature=args.temperature,
        top_p=args.top_p,
        logprobs=20,
    )

    _, response_objects = get_response_texts(
        model, tokenizer, conversations, sampling_params
    )

    logger.info("Extracting probabilities from completions")
    if isinstance(model, LLM):
        probabilities = calculate_probabilities(
            response_objects, tokenizer, target_words=["1", "2", "3", "4", "5"]
        )
    else:
        probabilities = calculate_probabilities_openai(
            response_objects, target_words=["1", "2", "3", "4", "5"]
        )

    logger.info("Grouping probabilities by completion")
    n_aspects = len(ASPECT2ANNOTATION_PROMPT)
    grouped_probabilities = [
        probabilities[i : i + n_aspects]
        for i in range(0, len(probabilities), n_aspects)
    ]

    logger.info("Calculating scores for each completion")
    scores = [grouped_probabilities_to_score(probs) for probs in grouped_probabilities]

    assert len(scores) == len(unrolled_dataset)

    # Create a new dataset with only the 'id' column
    out_dataset = Dataset.from_dict({"text": text})
    out_dataset = out_dataset.add_column("subset", subsets)
    out_dataset = out_dataset.add_column("id", ids)
    out_dataset = out_dataset.add_column("scores", scores)

    out_dataset = reroll_and_score_dataset(
        out_dataset, total_completions, cols_to_combine=["text", "scores"]
    )
    out_dataset = out_dataset.add_column("num_correct", num_correct)

    # get core dataset
    results_grouped = {}
    model_name = args.model_path
    results_grouped["model"] = model_name
    chat_template = (
        "None" if not check_tokenizer_chat_template(tokenizer) else "tokenizer"
    )
    results_grouped["chat_template"] = chat_template

    # print per subset and log into results_grouped file
    results_file = open(os.path.join(args.output_path, "results.txt"), "w")
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        # recompute "results" column for ties subset with different scoring method
        if subset.lower() == "ties":
            ties_subset_with_results, overall_score = process_single_model(
                subset_dataset
            )
            subset_dataset = ties_subset_with_results

            # Update the results for the ties subset in the original dataset
            ties_indices = [
                i for i, s in enumerate(out_dataset["subset"]) if s == "ties"
            ]
            out_dataset_df = out_dataset.to_pandas()
            for i, ties_idx in enumerate(ties_indices):
                out_dataset_df.at[ties_idx, "results"] = ties_subset_with_results[
                    "results"
                ][i]
            out_dataset = Dataset.from_pandas(out_dataset_df)

            print(f"{subset}: Overall score {overall_score}")
            results_file.write(f"{subset}: Overall score {overall_score}\n")
            results_grouped[subset] = overall_score
        else:
            num_correct = sum(subset_dataset["results"])
            num_total = len(subset_dataset["results"])
            print(f"{subset}: {num_correct}/{num_total} ({num_correct / num_total})")
            results_file.write(
                f"{subset}: {num_correct}/{num_total} ({num_correct / num_total})\n"
            )
            results_grouped[subset] = num_correct / num_total

    # Save the first sample to JSON
    with open(os.path.join(args.output_path, "first_sample.json"), "w") as f:
        json.dump(out_dataset[0], f, indent=2)

    logger.info("Saving output dataset")
    out_dataset.save_to_disk(args.output_path)


if __name__ == "__main__":
    args = parse_args()

    output_dir_exists = os.path.exists(args.output_path)
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("Logging into HuggingFace")
    setup(login_to_hf=True)

    if args.seed:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    # This is hardcoded for https://huggingface.co/datasets/allenai/reward-bench-2
    logger.info("Loading RewardBench dataset")
    reward_bench_dataset = load_dataset("allenai/reward-bench-2")
    reward_bench_dataset = reward_bench_dataset["test"]

    if args.debug:
        logger.info("Debug mode: Only annotating completions the first 10 prompts")
        reward_bench_dataset = reward_bench_dataset.select(range(10))

    logger.info(f"Annotating {len(reward_bench_dataset)} samples")
    run(reward_bench_dataset, args)
