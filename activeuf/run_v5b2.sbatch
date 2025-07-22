#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --environment=activeuf_dev
#SBATCH --time=00:45:00
#SBATCH --output=run_%j.out
#SBATCH --exclude=nid006438,nid006439,nid006440,nid006441,nid006442,nid006443,nid006444,nid006445,nid006446,nid006447,nid006448,nid006449,nid006450,nid006451,nid006461,nid006462,nid006868

python -m activeuf.get_raw_annotations_v5b \
        --dataset_path HuggingFaceH4/ultrafeedback_binarized \
        --model_name="meta-llama/Llama-3.3-70B-Instruct" \
        --max_tokens 24000 \
        --output_path /iopsstor/scratch/cscs/dmelikidze/datasets/raw_annotations4_llama_rank \
        --model_class vllm \
        --temperature 0.0 \
        --top_p 0.1 \
        --debug