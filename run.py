import argparse
import os

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--part", type=int)
	args = parser.parse_args()
	os.system(
		f"python -m activeuf.annotate_completions \
			--dataset_path datasets/allenai/ultrafeedback_binarized_cleaned/train_prefs-with-completions-sanitized \
			--model_name meta-llama/Llama-3.2-1B-Instruct \
			--download_dir /iopsstor/scratch/cscs/jessica/hf_cache \
			--part {args.part} --debug"
	)
