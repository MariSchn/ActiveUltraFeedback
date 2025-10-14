import random

from collections import defaultdict
from datasets import load_from_disk

if __name__ == "__main__":
    dataset = load_from_disk("/iopsstor/scratch/cscs/smarian/datasets/3_merged_completions/skywork_with_small")
    dataset = dataset.select(random.sample(range(len(dataset)), 5))
    output_file = open("example_completions.md", "w")
    
    model_to_completions = defaultdict(list)
    for sample in dataset:
        for completion in sample["completions"]:
            model_to_completions[completion["model"]].append(completion["response_text"])

    prompts = []
    for sample in dataset:
        prompts.append(sample["prompt"])

    output_file.write("# Example Completions\n\n")

    output_file.write("## Prompts\n\n")
    for i, prompt in enumerate(prompts):
        output_file.write(f"<details>\n\n<summary>Prompt {i+1}</summary>\n\n")
        output_file.write(f"``````\n{prompt}\n``````\n\n")
        output_file.write(f"</details>\n\n")

    for model in sorted(model_to_completions.keys()):
        completions = model_to_completions[model]
        output_file.write(f"## Model: {model}\n\n")
        for i, completion in enumerate(completions):
            output_file.write(f"<details>\n\n<summary>Completion {i+1}</summary>\n\n")
            output_file.write(f"``````\n{completion}\n``````\n\n")
            output_file.write(f"</details>\n\n")

    output_file.close()