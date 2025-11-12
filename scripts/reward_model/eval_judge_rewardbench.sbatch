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

# ===== CONFIGURATION VARIABLES =====
export OUTPUT_PATH="<OUTPUT_PATH>"
export MODEL_PATH="<MODEL_PATH>"

# ===== EVALUATION =====
python -m scripts.reward_model.eval_judge_rewardbench \
    --model_path $MODEL_PATH \
    --output_path $OUTPUT_PATH \
    --max_tokens 128 \
    --model_class vllm_server \
    --temperature 0.0 \
    --top_p 0.1
