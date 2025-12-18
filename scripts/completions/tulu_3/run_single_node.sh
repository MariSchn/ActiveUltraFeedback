#!/bin/bash

CACHE_DIR="$SCRATCH/cache"
DATASETS_DIR="$SCRATCH/datasets"

# Run configs:
# Format: MODEL:SEED:CHUNK_INDEX:NUM_CHUNKS
#   MODEL: model path
#   SEED: seed for this specific job
#   CHUNK_INDEX: index of this chunk (0-indexed), use -1 if not chunked
#   NUM_CHUNKS: total number of chunks the dataset is split into, use 0 if not chunked
CONFIGS=(
"CohereLabs/c4ai-command-a-03-2025:15:0:2"
"CohereLabs/c4ai-command-a-03-2025:15:1:2"
# "deepseek-ai/DeepSeek-V3:16:-1:0"  # Too big for single node
"google/gemma-3-1b-it:46:-1:0"
"google/gemma-3-4b-it:47:-1:0"
"google/gemma-3-12b-it:13:-1:0"
"google/gemma-3-27b-it:14:-1:0"
# "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:12:-1:0"  # Too big for single node
"nvidia/Llama-3_3-Nemotron-Super-49B-v1:10:-1:0"
"meta-llama/Llama-3.1-8B-Instruct:5:-1:0"
"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF:11:0:2"
"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF:11:1:2"
"nvidia/Llama-3.1-Nemotron-Nano-8B-v1:44:-1:0"
"allenai/Llama-3.1-Tulu-3-70B:18:-1:0"
# "allenai/Llama-3.1-Tulu-3-405B:19:-1:0"  # Too big for a single node
"meta-llama/Llama-3.2-1B-Instruct:52:-1:0"
"meta-llama/Llama-3.2-3B-Instruct:53:-1:0"
"meta-llama/Llama-3.3-70B-Instruct:6:-1:0"
"mistralai/Mistral-Large-Instruct-2411:9:0:2"
"mistralai/Mistral-Large-Instruct-2411:9:1:2"
"mistralai/Mistral-Small-24B-Instruct-2501:8:-1:0"
"moonshotai/Moonlight-16B-A3B-Instruct:20:-1:0"
"allenai/OLMo-2-0325-32B-Instruct:17:-1:0"
"microsoft/phi-4:7:-1:0"
"microsoft/Phi-4-mini-instruct:51:-1:0"
"Qwen/Qwen2.5-0.5B-Instruct:43:-1:0"
"Qwen/Qwen2.5-72B-Instruct:3:0:2"
"Qwen/Qwen2.5-72B-Instruct:3:1:2"
"Qwen/Qwen3-0.6B:48:-1:0"
"Qwen/Qwen3-1.7B:49:-1:0"
"Qwen/Qwen3-14B:0:-1:0"
"Qwen/Qwen3-30B-A3B:1:-1:0"
"Qwen/Qwen3-32B:2:-1:0"
# "Qwen/Qwen3-235B-A22B:4:-1:0"  # Too big for single node
"HuggingFaceTB/SmolLM2-1.7B-Instruct:42:-1:0"
"HuggingFaceTB/SmolLM2-135M-Instruct:45:-1:0"
)

for config in "${CONFIGS[@]}"; do
    # Split the config
    IFS=':' read -r MODEL SEED CHUNK_IDX NUM_CHUNKS <<< "$config"
    MODEL_NAME="${MODEL##*/}"

    # Determine job name, output path, and log path based on whether chunking is used
    if [ "$CHUNK_IDX" -eq -1 ]; then
        JOB_NAME="${MODEL_NAME}"
        OUTPUT_PATH="${DATASETS_DIR}/2_full_completions/tulu_3/${MODEL_NAME}"
        LOG_PATH="./logs/completions/tulu_3/${MODEL_NAME}/%j.out"
    else
        JOB_NAME="${MODEL_NAME}_${CHUNK_IDX}"
        OUTPUT_PATH="${DATASETS_DIR}/1_partial_completions/tulu_3/${JOB_NAME}"
        LOG_PATH="./logs/completions/tulu_3/${MODEL_NAME}/chunk_${CHUNK_IDX}_%j.out"
    fi

    echo "Submitting job for model: $MODEL_NAME (chunk: $CHUNK_IDX/$NUM_CHUNKS) with seed: $SEED"

    # Build optional arguments for chunking
    OPTIONAL_ARGS=""
    if [ "$CHUNK_IDX" -ne -1 ]; then
        OPTIONAL_ARGS="--num_chunks $NUM_CHUNKS --chunk_index $CHUNK_IDX"
    fi
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --container-writable
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$LOG_PATH

export HF_HOME=${CACHE_DIR}/hf_cache
export WANDB_DIR=${CACHE_DIR}/wandb
export TRANSFORMERS_CACHE=${CACHE_DIR}/transformers
export HF_DATASETS_CACHE=${CACHE_DIR}/datasets
export TORCH_HOME=${CACHE_DIR}/torch
export XDG_CACHE_HOME=${CACHE_DIR}
export TORCH_EXTENSIONS_DIR=${XDG_CACHE_HOME}/torch_extensions

# Note: Mistral Models and MoE models need the activeuf_dev container (given in Dockerfile)
srun --environment=activeuf_new_xformers python -u -m activeuf.completions.generate_completions \
    --dataset_path ${DATASETS_DIR}/0_raw_datasets/llama-3.1-tulu-3-8b-preference-mixture/ \
    --model_name $MODEL \
    --model_class vllm \
    --output_path $OUTPUT_PATH \
    --seed $SEED \
    $OPTIONAL_ARGS
EOF
done