#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --container-writable
#SBATCH --job-name=Llama-3.3-70B-Instruct_0
#SBATCH --output=./logs/swiss_ai/Llama-3.3-70B-Instruct_0_%j.out
#SBATCH --exclude=nid006438,nid006439,nid006440,nid006441,nid006442,nid006443,nid006444,nid006445,nid006446,nid006447,nid006448,nid006449,nid006450,nid006451,nid006461,nid006462,nid006868

export HF_TOKEN=$(cat ~/.hf-token)
dataset_path="/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/olmo-2-0325-32b-preference-mix-promptsOnly"
model_name="meta-llama/Llama-3.3-70B-Instruct"

srun --environment=activeuf_dev python swiss_ai.py \
  --dataset_path "${dataset_path}" \
  --dataset_split "train" \
  --model_name "${model_name}" \
  --output_path "${dataset_path}/completions/${model_name#*/}_0.jsonl" \
  --num_chunks 2 \
  --chunk_index 0 