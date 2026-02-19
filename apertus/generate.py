"""
Lightweight script to send requests to a previously started OpenAI API server.
To start a server, see https://github.com/swiss-ai/model-launch.

Usage:
    python generate.py --dataset-path allenai/Dolci-Instruct-DPO --output-dir ./datasets/dolci_qwen3_8b --model Qwen/Qwen3-8B --base-url http://172.28.43.156:8080/v1
"""

import os
import json
import asyncio
import argparse

from openai import AsyncOpenAI
from dotenv import load_dotenv
from datasets import load_from_disk, load_dataset, DatasetDict
from tqdm.asyncio import tqdm_asyncio

load_dotenv()


async def main(args):
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=os.environ.get("CSCS_SERVING_KEY"),
    )

    if os.path.exists(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path)
        if isinstance(dataset, DatasetDict):
            dataset = dataset["train"]

    dataset = dataset.select(range(100))

    # Get prompts (everything except the last message)
    dataset = dataset.map(lambda x: {"prompt": x["chosen"][:-1]})

    semaphore = asyncio.Semaphore(1000)

    async def get_response(prompt):
        async with semaphore:
            res = await client.chat.completions.create(
                model=args.model,
                messages=prompt,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return res.choices[0].message.content

    tasks = [get_response(p) for p in dataset["prompt"]]
    responses = await tqdm_asyncio.gather(*tasks, desc="Generating")

    dataset = dataset.add_column("response", responses)

    output_dir = args.output_dir
    dataset.save_to_disk(output_dir)

    # Save the first sample as JSON to the output_dir
    with open(
        os.path.join(output_dir, "first_sample.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(dataset[0], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--base-url", type=str, default="https://api.swissai.cscs.ch/v1"
    )
    args = parser.parse_args()

    asyncio.run(main(args))
