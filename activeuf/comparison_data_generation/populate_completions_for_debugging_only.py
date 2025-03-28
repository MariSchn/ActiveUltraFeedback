import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to load (e.g. truthful_qa)")
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset_name}")
    load_path = f"./completion_data/{args.dataset_name}.json"
    dataset = json.load(open(load_path, "r"))

    
