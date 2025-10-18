MODELS=(
# "Qwen/Qwen3-14B"
# "Qwen/Qwen3-30B-A3B"
# "Qwen/Qwen3-32B"
# "Qwen/Qwen2.5-72B-Instruct"
# "Qwen/Qwen3-235B-A22B"  # Too big for single node
# "meta-llama/Llama-3.1-8B-Instruct"
# "meta-llama/Llama-3.3-70B-Instruct"
# "microsoft/phi-4"
# "mistralai/Mistral-Small-24B-Instruct-2501"
# "mistralai/Mistral-Large-Instruct-2411"
# "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
# "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
# "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"  # Too big for single node
# "google/gemma-3-12b-it"
# "google/gemma-3-27b-it"
# "CohereLabs/c4ai-command-a-03-2025"
# "deepseek-ai/DeepSeek-V3"  # Too big for single node
# "allenai/OLMo-2-0325-32B-Instruct"
# "allenai/Llama-3.1-Tulu-3-70B"
# "allenai/Llama-3.1-Tulu-3-405B"  # Too big for a single node
# "moonshotai/Moonlight-16B-A3B-Instruct"
# "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# "Qwen/Qwen2.5-0.5B-Instruct"
"nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
# "HuggingFaceTB/SmolLM2-135M-Instruct"
# google/gemma-3-1b-it
# google/gemma-3-4b-it
# Qwen/Qwen3-0.6B
# Qwen/Qwen3-1.7B
# HuggingFaceTB/SmolLM3-3B
# microsoft/Phi-4-mini-instruct
# meta-llama/Llama-3.2-1B-Instruct
# meta-llama/Llama-3.2-3B-Instruct
)

# Define a SEEDS array matching the MODELS order
SEEDS=(
# 0
# 1
# 2
# 3
# "Qwen/Qwen3-235B-A22B"  # Too big for single node
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"  # Too big for single node
# 11
# 12
# 13
# 14
# 15
# 16
# "allenai/Llama-3.1-Tulu-3-405B"  # Too big for a single node
# 17
# 42
# 43
44
# 45
# 46
# 47
# 48
# 49
# 50
# 51
# 52
# 53
)

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NAME="${MODEL##*/}"
    SEED="${SEEDS[$i]}"


    echo "Submitting job for model: $MODEL_NAME ($MODEL) with seed: $SEED"
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --container-writable
#SBATCH --job-name=$MODEL_NAME
#SBATCH --output=./logs/completions/$MODEL_NAME/%j.out

export HF_HOME=/iopsstor/scratch/cscs/smarian/hf_cache
export WANDB_DIR=/iopsstor/scratch/cscs/smarian/wandb
export TRANSFORMERS_CACHE=/iopsstor/scratch/cscs/smarian/.cache/transformers
export HF_DATASETS_CACHE=/iopsstor/scratch/cscs/smarian/.cache/datasets
export TORCH_HOME=/iopsstor/scratch/cscs/smarian/.cache/torch
export XDG_CACHE_HOME=/iopsstor/scratch/cscs/smarian/.cache/.cache
export TORCH_EXTENSIONS_DIR=$XDG_CACHE_HOME/torch_extensions

srun --environment=activeuf_new_xformers python -u -m activeuf.generate_completions \
    --dataset_path /iopsstor/scratch/cscs/smarian/datasets/0_raw_datasets/ultrafeedback_binarized_cleaned/train_prefs \
    --model_name $MODEL \
    --model_class vllm \
    --output_path /iopsstor/scratch/cscs/smarian/datasets/2_full_completions/ultrafeedback/$MODEL_NAME \
    --seed $SEED
EOF
done