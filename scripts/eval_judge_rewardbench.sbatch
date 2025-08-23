#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --account=a-infra01-1
#SBATCH --exclusive
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --partition=normal
#SBATCH --time=1:00:00
#SBATCH --container-writable
#SBATCH --job-name=judge_rewardbench
#SBATCH --output=./logs/judge/rewardbench_%j.out
#SBATCH --environment=activeuf_dev

python -m scripts.eval_judge_rewardbench \
    --model_path="Qwen/Qwen3-32B" \
    --output_path /iopsstor/scratch/cscs/smarian/data/judge_rewardbench/Qwen3-32B \
    --max_tokens 128 \
    --model_class vllm_server \
    --temperature 0.0 \
    --top_p 0.1
