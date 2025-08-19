#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=debug
#SBATCH --time=1:00:00
#SBATCH --container-writable
#SBATCH --job-name=judge_rewardbench
#SBATCH --output=./logs/judge/rewardbench_%j.out
#SBATCH --error=./logs/judge/rewardbench_%j.err
#SBATCH --exclude=nid006438,nid006439,nid006440,nid006441,nid006442,nid006443,nid006444,nid006445,nid006446,nid006447,nid006448,nid006449,nid006450,nid006451,nid006461,nid006462,nid006868,nid006476,nid005557,nid006455,nid007122,nid007119,nid006513,nid005813,nid006454,nid006452,nid006457,nid005230,nid005248,nid007117
#SBATCH --environment=activeuf_dev

pip install resources/reward-bench

python -m scripts.eval_judge_rewardbench \
    --model_path="Qwen/Qwen3-32B" \
    --output_path /iopsstor/scratch/cscs/smarian/data/judge_rewardbench/Qwen3-32B \
    --max_tokens 24000 \
    --model_class vllm \
    --temperature 0.0 \
    --top_p 0.1

python -m scripts.eval_judge_rewardbench \
    --model_path="meta-llama/Llama-3.3-70B-Instruct" \
    --output_path /iopsstor/scratch/cscs/smarian/data/judge_rewardbench/Llama-3.3-70B-Instruct \
    --max_tokens 24000 \
    --model_class vllm \
    --temperature 0.0 \
    --top_p 0.1