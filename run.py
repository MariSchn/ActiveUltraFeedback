import argparse
import os

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--part", type=int)
	args = parser.parse_args()
	os.system(
		f"python -m activeuf.bulk_annotate \
			--inputs_path datasets/inputs_for_bulk_annotation \
			--model_name meta-llama/Llama-3.3-70B-Instruct \
			--download_dir ./hf_cache \
			--part_size 1000 \
			--part {args.part}"
	)
