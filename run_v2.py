import os

if __name__ == "__main__":
	os.system(
		f"python -m activeuf.get_raw_annotations_v2 \
			--dataset_path datasets/merged_completions \
			--model_name meta-llama/Llama-3.3-70B-Instruct \
			--max_tokens 24000 \
			--download_dir ./hf_cache \
			--output_path datasets/raw_annotations_v2 \
			--debug"
	)
