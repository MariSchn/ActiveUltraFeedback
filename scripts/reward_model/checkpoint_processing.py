import os
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Process checkpoints directory.")
parser.add_argument(
    "--checkpoints_dir", type=str, required=True, help="Path to checkpoints directory"
)
parser.add_argument(
    "--output_txt", type=str, required=True, help="Path to output text file"
)
args = parser.parse_args()

base_model_name = "allenai/Llama-3.1-Tulu-3-8B-SFT"
checkpoints_dir = args.checkpoints_dir
output_suffix = "-processed"

processed_list = []
for folder in tqdm(os.listdir(checkpoints_dir)):
    if not folder.startswith("checkpoint-") or folder.endswith("-processed"):
        continue

    checkpoint_path = os.path.join(checkpoints_dir, folder)
    output_path = os.path.join(checkpoints_dir, folder + output_suffix)

    # Skip if processed directory already exists
    if os.path.exists(output_path):
        print(f"Processed directory {output_path} already exists. Skipping.")
        processed_list.append(output_path)
        continue

    print(f"Processing {checkpoint_path} -> {output_path}")

    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        print(
            "No pad_token present. Assigning tokenizer.pad_token = tokenizer.eos_token"
        )
        tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(adapter_config_path):
        print("Found LoRA adapter, merging...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        merged_model = model.merge_and_unload()
    else:
        print("No LoRA adapter, loading model directly...")
        merged_model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=1,
            pad_token_id=tokenizer.pad_token_id,
        )

    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved processed checkpoint to {output_path}")

    processed_list.append(output_path)

with open(args.output_txt, "w") as f:
    for path in processed_list:
        f.write(path + "\n")
