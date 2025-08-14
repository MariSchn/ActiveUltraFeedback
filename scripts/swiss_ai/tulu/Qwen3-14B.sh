#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --container-writable
#SBATCH --job-name=Qwen3-14B
#SBATCH --output=./logs/swiss_ai/tulu/Qwen3-14B_%j.out
#SBATCH --exclude=nid006438,nid006439,nid006440,nid006441,nid006442,nid006443,nid006444,nid006445,nid006446,nid006447,nid006448,nid006449,nid006450,nid006451,nid006461,nid006462,nid006868

# export HF_TOKEN=$(cat ~/.hf-token)
dataset_path="/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/llama-3.1-tulu-3-70b-preference-mixture"
model_name="Qwen/Qwen3-14B"

srun --environment=activeuf_dev python swiss_ai.py \
  --dataset_path "${dataset_path}" \
  --dataset_split "train" \
  --model_name "${model_name}" \
  --output_path "${dataset_path}/completions/${model_name#*/}.jsonl" \
  --seed 12