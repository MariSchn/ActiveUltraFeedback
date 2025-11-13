#!/bin/bash

CACHE_DIR="$SCRATCH/cache"
DATASETS_DIR="$SCRATCH/datasets"

CONFIGS=(
"Qwen/Qwen3-14B:0"
"Qwen/Qwen3-30B-A3B:1"
"Qwen/Qwen3-32B:2"
"Qwen/Qwen2.5-72B-Instruct:3"
# "Qwen/Qwen3-235B-A22B:4"  # Too big for single node
"meta-llama/Llama-3.1-8B-Instruct:5"
"meta-llama/Llama-3.3-70B-Instruct:6"
"microsoft/phi-4:7"
"mistralai/Mistral-Small-24B-Instruct-2501:8"
"mistralai/Mistral-Large-Instruct-2411:9"
"nvidia/Llama-3_3-Nemotron-Super-49B-v1:10"
"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF:11"
# "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:12"  # Too big for single node
"google/gemma-3-12b-it:13"
"google/gemma-3-27b-it:14"
"CohereLabs/c4ai-command-a-03-2025:15"
# "deepseek-ai/DeepSeek-V3:16"  # Too big for single node
"allenai/OLMo-2-0325-32B-Instruct:17"
"allenai/Llama-3.1-Tulu-3-70B:18"
# "allenai/Llama-3.1-Tulu-3-405B:19"  # Too big for a single node
"moonshotai/Moonlight-16B-A3B-Instruct:20"
"HuggingFaceTB/SmolLM2-1.7B-Instruct:42"
"Qwen/Qwen2.5-0.5B-Instruct:43"
"nvidia/Llama-3.1-Nemotron-Nano-8B-v1:44"
"HuggingFaceTB/SmolLM2-135M-Instruct:45"
"google/gemma-3-1b-it:46"
"google/gemma-3-4b-it:47"
"Qwen/Qwen3-0.6B:48"
"Qwen/Qwen3-1.7B:49"
"HuggingFaceTB/SmolLM3-3B:50"
"microsoft/Phi-4-mini-instruct:51"
"meta-llama/Llama-3.2-1B-Instruct:52"
"meta-llama/Llama-3.2-3B-Instruct:53"
)

for pair in "${CONFIGS[@]}"; do
    # Split the pair into MODEL and SEED
    MODEL="${pair%:*}"
    SEED="${pair##*:}"
    MODEL_NAME="${MODEL##*/}"


    echo "Submitting job for model: $MODEL_NAME ($MODEL) with seed: $SEED"
    
    sbatch <<EOF
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

export HF_HOME=${CACHE_DIR}/hf_cache
export WANDB_DIR=${CACHE_DIR}/wandb
export TRANSFORMERS_CACHE=${CACHE_DIR}/transformers
export HF_DATASETS_CACHE=${CACHE_DIR}/datasets
export TORCH_HOME=${CACHE_DIR}/torch
export XDG_CACHE_HOME=${CACHE_DIR}/.cache
export TORCH_EXTENSIONS_DIR=${XDG_CACHE_HOME}/torch_extensions

srun --environment=activeuf_new_xformers python -u -m activeuf.generate_completions \
    --dataset_path ${DATASETS_DIR}/0_raw_datasets/ultrafeedback_binarized_cleaned/train_prefs \
    --model_name $MODEL \
    --model_class vllm \
    --output_path ${DATASETS_DIR}/2_full_completions/ultrafeedback/$MODEL_NAME \
    --seed $SEED
EOF
done