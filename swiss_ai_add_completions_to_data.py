import datasets
import os
import argparse
import json

if __name__ == "__main__":
    """
    Collecting the completions and adding them to the dataset and conversation branches
    Example command
    python swiss_ai_add_completions_to_data.py \
        --dataset_path  /capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/olmo-2-0325-32b-preference-mix-promptsOnly  \
        --output_path /capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/olmo-2-0325-32b-preference-mix-newCompletions
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        type=str,
    )
    args = parser.parse_args()

    print("Load dataset")
    dataset = datasets.load_from_disk(args.dataset_path)

    print("Finding list of jsonl completions")
    completion_paths = os.path.join(args.dataset_path, "completions")
    completion_name_list = [x for x in os.listdir(completion_paths) if x.endswith(".jsonl") and not (x.endswith("_0.jsonl") or x.endswith("_1.jsonl"))]
    print("Model completions found: ", completion_name_list)

    for completion_name in completion_name_list:
        print("Processing {}".format(completion_name))
        completions_data = {}
        with open(os.path.join(completion_paths, completion_name), "r") as f:
            for row in f:
                row = json.loads(row)
                row_completion = row["completion"]
                completions_data[row["conversation_id"]] = {
                    "model": row_completion["model"],
                    "principle": row_completion["principle"] if row_completion["messages"][0]["role"] == "system" else None,
                    "system_prompt": row_completion["messages"][0]["content"] if row_completion["messages"][0]["role"] == "system" else None,
                    "completion": row_completion["response_text"]
                }

        dataset["train"] = dataset["train"].map(
            lambda x: {
                "conversation_branches": x["conversation_branches"] + [
                    {
                        "messages": [
                            {
                                "role": "assistant",
                                "parts": [
                                    {
                                        "type": "response",
                                        "content": completions_data[x["conversation_id"]]["completion"],
                                        "metadata": {
                                            "model": completions_data[x["conversation_id"]]["model"],
                                            "principle": completions_data[x["conversation_id"]]["principle"],
                                            "system_prompt": completions_data[x["conversation_id"]]["system_prompt"],
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        )

    print("First entry of the processed dataset.")
    print(dataset["train"][0])
    print("Saving to: ", args.output_path)
    dataset.save_to_disk(args.output_path)
