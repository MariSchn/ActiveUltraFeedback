import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_model",
        type=str,
        required=False,
        default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    )
    parser.add_argument("--model_to_annotate", type=str, required=True)
    parser.add_argument("--input_path")
    parser.add_argument("--output_path", type=str, required=False, default=None)
    args = parser.parse_args()

    # Load model and tokenizer
    model_name = args.annotation_model
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    dataset = load_from_disk(args.input_path)

    # Remove all completions except for the one to annotate
    def filter_completions(example):
        example["completions"] = [
            c for c in example["completions"] if c["model"] == args.model_to_annotate
        ]
        return example

    dataset = dataset.map(filter_completions)

    # Annotate
    def annotate(example):
        model_completion = example["completions"][0]

        conv = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": model_completion["response_text"]},
        ]
        conv_formatted = tokenizer.apply_chat_template(conv, tokenize=False)
        if tokenizer.bos_token is not None and conv_formatted.startswith(
            tokenizer.bos_token
        ):
            conv_formatted = conv_formatted[len(tokenizer.bos_token) :]
        conv_tokenized = tokenizer(conv_formatted, return_tensors="pt")
        with torch.no_grad():
            score = rm(**conv_tokenized).logits[0][0].item()
        model_completion["overall_score"] = str(score)

        return example

    dataset = dataset.map(
        annotate,
    )
    dataset.save_to_disk(f"{args.output_path}/{args.model_to_annotate.split('/')[-1]}")
    with open(
        f"{args.output_path}/{args.model_to_annotate.split('/')[-1]}/first_sample.json",
        "w",
    ) as f:
        json.dump(dataset[0], f, indent=2)
