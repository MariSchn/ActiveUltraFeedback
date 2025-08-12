import huggingface_hub
import os
import vllm
import argparse

from datasets import load_dataset, load_from_disk, concatenate_datasets
from dotenv import load_dotenv

from activeuf.utils import setup, load_model, get_response_texts
import json


def concat_datasets(dataset_base_path):
    output_path = f"/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/full/{dataset_base_path.split('/')[-1]}"

    # Load datasets from disk
    dataset1 = load_from_disk(f"{dataset_base_path}_1")
    dataset2 = load_from_disk(f"{dataset_base_path}_2")

    # Merge datasets
    merged_dataset = concatenate_datasets([dataset1, dataset2])

    # Save merged dataset to disk
    merged_dataset.save_to_disk(output_path)

def remove_format(dataset_path):
    dataset = load_from_disk(dataset_path)

    model_name = dataset_path.split("/")[-1]
    out_path = f"/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/final/{model_name}"

    with_principles = ("principles" in model_name or "w_principles" in model_name)

    print(f"Processing dataset: {dataset_path}")
    print(f"With principles: {with_principles}")
    print(f"Output path: {out_path}")

    def format_sample(sample):

        out = {
            "id": sample["id"],
            "source": sample["source"],
            "model": sample["completion"]["model"],
        }

        out["principle"] = sample["completion"]["principle"] if with_principles else None

        messages = sample["completion"]["messages"]
        messages.append({
            "role": "assistant",
            "content": sample["completion"]["response_text"]
        })

        out["messages"] = messages

        return out
        
    formatted_dataset = dataset.map(
        format_sample,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )

    formatted_dataset.save_to_disk(out_path)
    dataset.to_json(f"{out_path}.json", orient="records", lines=True, force_ascii=False)

if __name__ == "__main__":
    # to_merge = [
    #     "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/raw/Llama-3.3-70B-Instruct",
    #     "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/raw/Llama-3.3-70B-Instruct_w_principles",
    #     "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/raw/Qwen2.5-72B-Instruct",
    #     "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/raw/Qwen2.5-72B-Instruct_w_principles",
    #     "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/raw/Qwen3-32B",
    #     "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/raw/Qwen3-32B_w_principles",
    # ]

    # for dataset_base_path in to_merge:
    #     concat_datasets(dataset_base_path)




    full_datasets = [
        "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/full/Llama-3.1-8B-Instruct",
        "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/full/Llama-3.1-8B-Instruct_w_principles",
        "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/full/Llama-3.3-70B-Instruct",
        "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/full/Llama-3.3-70B-Instruct_w_principles",
        "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/full/Qwen2.5-72B-Instruct",
        "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/full/Qwen2.5-72B-Instruct_w_principles",
        "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/full/Qwen3-14B",
        "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/full/Qwen3-14B_w_principles",
        "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/full/Qwen3-32B",
        "/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/full/Qwen3-32B_w_principles",
    ]

    for dataset_path in full_datasets:
        remove_format(dataset_path)





    sample = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/final/Llama-3.1-8B-Instruct")[0]
    with open("first_sample.json", "w") as f:
        json.dump(sample, f, indent=2)

    sample_with_principles = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/swiss_ai/final/Llama-3.1-8B-Instruct_w_principles")[0]
    with open("first_sample_with_principles.json", "w") as f:
        json.dump(sample_with_principles, f, indent=2)