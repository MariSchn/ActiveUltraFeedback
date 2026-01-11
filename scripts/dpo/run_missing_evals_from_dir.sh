#!/bin/bash

# ==============================================================================
# 1. Global Cache Configuration
# ==============================================================================
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

# Redirect caches to SCRATCH
export XDG_CACHE_HOME="${SCRATCH}/cache"
export HF_HOME="${SCRATCH}/huggingface"
export WANDB_DIR="${SCRATCH}/cache/wandb"
export WANDB_CACHE_DIR="${SCRATCH}/cache/wandb"
export TRITON_CACHE_DIR="${SCRATCH}/cache/triton"
export CUDA_CACHE_PATH="${SCRATCH}/cache/nv_compute"
export VLLM_CACHE_ROOT="${SCRATCH}/cache/vllm"
export PYTHONUSERBASE="${SCRATCH}/cache/python_user_base"

# Global Benchmark Definitions
BENCHMARK_FILES=(
    "results/gsm8k_tulu/metrics.json"
    "results/ifeval_tulu/metrics.json"
    # "results/minerva_math_tulu/metrics.json"
    "results/truthfulqa_tulu/metrics.json"
    "results/alpaca_eval/activeuf/leaderboard.csv"
)

# Parse ARGS
MODELS_BASE_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --models_dir)
            MODELS_BASE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --models_dir <path/to/folder/containing/models>"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODELS_BASE_DIR" ]]; then
    echo "Error: Please provide the models directory."
    echo "Usage: $0 --models_dir <path>"
    exit 1
fi

if [[ ! -d "$MODELS_BASE_DIR" ]]; then
    echo "Error: Directory does not exist: $MODELS_BASE_DIR"
    exit 1
fi

# ==============================================================================
# Helper: Check for valid model files (Borrowed from Script 2)
# ==============================================================================
is_valid_model_dir() {
    local dir="$1"
    # Check for existence of any weight file type in the immediate directory
    local has_weights=$(find "$dir" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \) -print -quit)

    if [ -n "$has_weights" ]; then
        return 0 # Valid
    else
        return 1 # Invalid
    fi
}

# ==============================================================================
# Helper Function: Launch Generic Evaluation Job
# ==============================================================================
launch_model_eval() {
    local model_dir=$1
    local benchmark=$2
    local task_name=$3
    local run_id=$(basename "$model_dir")

    echo "  Processing: $run_id"
    echo "    Benchmark: $benchmark"
    echo "    Task: $task_name"
    
    # Create results directory inside the model folder
    mkdir -p "${model_dir}/results/${benchmark}"
    
    # Submit Evaluation job
    echo "    Submitting evaluation job..."
    sbatch --job-name="${run_id}_${benchmark}" \
           --account="a-infra01-1" \
           --output="${model_dir}/results/${benchmark}/eval_%j.log" \
           --nodes=1 \
           --ntasks=1 \
           --gpus-per-task=4 \
           --time=5:00:00 \
           --partition=normal \
           --exclude=nid[006845,006851,006854-006855,006752,006755,006787-006788,007095-007096,007098-007099,006824,006826,006845,006851,006854-006855,006752,006755,006787-006788,007095-007096,007098-007099,006708,006789-006791,007096,007098-007099,007104,006869,006875,007171,007173-007174,007182,007184,007201,007203,007189-007190,007234-007235,007087] \
           --wrap="
            # --- CACHE REDIRECTION (Inside Job) ---
            export XDG_CACHE_HOME=\"${SCRATCH}/cache\"
            export HF_HOME=\"${SCRATCH}/cache/hf_cache\"
            export TRITON_CACHE_DIR=\"${SCRATCH}/cache/triton\"
            export TORCH_EXTENSIONS_DIR=\"${SCRATCH}/cache/torch_extensions\"
            export TORCHINDUCTOR_CACHE_DIR=\"${SCRATCH}/cache/torch_inductor\"
            export CUDA_CACHE_PATH="${SCRATCH}/cache/nv_compute"
            export VLLM_CACHE_ROOT="${SCRATCH}/cache/vllm"
            export PYTHONUSERBASE="${SCRATCH}/cache/python_user_base"

            export VLLM_WORKER_MULTIPROC_METHOD=spawn
            export PROJECT_ROOT_AT=$SCRATCH/projects/ActiveUltraFeedback/resources/olmes
            export PROJECT_NAME=olmes
            export PACKAGE_NAME=oe_eval
            export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
            export SKIP_INSTALL_PROJECT=1
            export SHARED=/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared
            export OMP_NUM_THREADS=1
            export TOKENIZERS_PARALLELISM=false
            export CONTAINER_IMAGES=/capstor/store/cscs/swissai/infra01/container-images
            unset SSL_CERT_FILE
            export VLLM_DISABLE_COMPILE_CACHE=1
            
            CONTAINER_ARGS=\"\
              --container-image=\$CONTAINER_IMAGES/infra01+ismayilz+olmes+arm64-cuda-root-latest.sqsh \
              --environment=\${PROJECT_ROOT_AT}/installation/edf.toml \
              --container-mounts=\
\$PROJECT_ROOT_AT,\
${SCRATCH},\
/iopsstor/scratch/cscs/smoalla/projects/dpr/,\
\$SHARED,\
\$HOME/.gitconfig,\
\$HOME/.bashrc,\
\$HOME/.ssh \
              --container-workdir=\$PROJECT_ROOT_AT \
              --no-container-mount-home \
              --no-container-remap-root \
              --no-container-entrypoint \
              --container-writable \
              /opt/template-entrypoints/pre-entrypoint.sh\"
            
            EVAL_ARGS=\"\
              --model=${run_id} \
              --model-wb-name=${run_id} \
              --model-type=vllm \
              --batch-size=1 \
              --model-args '{\\\"tensor_parallel_size\\\": 4, \\\"max_length\\\": 4096, \\\"add_bos_token\\\": false, \\\"model_path\\\": \\\"${model_dir}\\\", \\\"trust_remote_code\\\": true}' \
              --use-chat-format=True\"
            
            # NOTE: Output dir is set to local model folder
            srun --nodes=1 --ntasks=1 --gpus-per-task=4 \$CONTAINER_ARGS bash -c \"exec python3 -m oe_eval.launch --task=${task_name} --output-dir=${model_dir}/results/${benchmark} \$EVAL_ARGS\"
            "
    
    echo "    Job submitted for ${run_id}_${benchmark}"
}

# ==============================================================================
# Helper Function: Launch Alpaca Eval
# ==============================================================================
launch_alpaca_eval() {
    local model_dir=$1
    local run_id=$(basename "$model_dir")

    local results_dir="${model_dir}/results/alpaca_eval"
    
    echo "  Processing: $run_id (Alpaca Eval)"
    
    mkdir -p "${results_dir}"
    
    echo "    Submitting Alpaca Eval job..."
    sbatch --job-name="${run_id}_alpaca_eval" \
           --account="a-infra01-1" \
           --output="${results_dir}_log/log_%j.out" \
           --error="${results_dir}_log/log_%j.err" \
           --nodes=1 \
           --ntasks=1 \
           --gpus-per-task=4 \
           --cpus-per-task=32 \
           --time=01:15:00 \
           --partition=normal \
           --environment=activeuf \
           --wrap="
               # --- CACHE REDIRECTION (Inside Job) ---
               export XDG_CACHE_HOME=\"${SCRATCH}/cache\"
               export HF_HOME=\"${SCRATCH}/cache/hf_cache\"
               export TRITON_CACHE_DIR=\"${SCRATCH}/cache/triton\"
               export VLLM_NO_USAGE_STATS=1
               export TORCH_EXTENSIONS_DIR=\"${SCRATCH}/cache/torch_extensions\"
               export TORCHINDUCTOR_CACHE_DIR=\"${SCRATCH}/cache/torch_inductor\"
               export CUDA_CACHE_PATH="${SCRATCH}/cache/nv_compute"
               export VLLM_CACHE_ROOT="${SCRATCH}/cache/vllm"
               export PYTHONUSERBASE="${SCRATCH}/cache/python_user_base"
               export VLLM_DISABLE_COMPILE_CACHE=1

               cd ${SCRATCH}/ActiveUltraFeedback
               pip install \"datasets<3.0.0\" --quiet
               export MODEL_PATH=\"${model_dir}\"
               export RESULTS_DIR=\"${results_dir}\"
               
               bash scripts/dpo/run_alpaca_eval.sh
           "
    echo "    Job submitted for ${run_id}_alpaca_eval"
}

# ==============================================================================
# Main Logic: Iterate and Check
# ==============================================================================

echo -e "\n===================================="
echo -e "===== CHECKING MODELS IN: $MODELS_BASE_DIR ====="
echo -e "====================================\n"

# Get all directories
model_dirs=()
for dir in "$MODELS_BASE_DIR"/*; do
    if [[ -d "$dir" ]]; then
        model_dirs+=("$dir")
    fi
done

echo "Found ${#model_dirs[@]} sub-directories."

# Tracking missing evals
declare -A missing_gsm8k
declare -A missing_ifeval
# declare -A missing_minerva_math
declare -A missing_truthfulqa
declare -A missing_alpaca

echo "--- Looking for missing evaluations... ---"

for full_model_path in "${model_dirs[@]}"; do
    dir_name=$(basename "$full_model_path")

    # 1. Validate if it is actually a model directory
    if ! is_valid_model_dir "$full_model_path"; then
        echo "  [SKIP] $dir_name (No model weights found)"
        continue
    fi

    # 2. Check for missing benchmarks
    for benchmark_file in "${BENCHMARK_FILES[@]}"; do
        full_result_path="$full_model_path/$benchmark_file"
        
        if [[ ! -f "$full_result_path" ]]; then
            if [[ "$benchmark_file" == *"gsm8k"* ]]; then
                missing_gsm8k["$full_model_path"]=1
            elif [[ "$benchmark_file" == *"ifeval"* ]]; then
                missing_ifeval["$full_model_path"]=1
            # elif [[ "$benchmark_file" == *"minerva_math"* ]]; then
            #     missing_minerva_math["$full_model_path"]=1
            elif [[ "$benchmark_file" == *"truthfulqa"* ]]; then
                missing_truthfulqa["$full_model_path"]=1
            elif [[ "$benchmark_file" == *"alpaca_eval"* ]]; then
                missing_alpaca["$full_model_path"]=1
            fi
        fi
    done
done


# ==============================================================================
# Launch Jobs
# ==============================================================================

# GSM8K
if [[ ${#missing_gsm8k[@]} -gt 0 ]]; then
    echo -e "\n--- Launching GSM8K ---"
    for model_path in "${!missing_gsm8k[@]}"; do
        launch_model_eval "$model_path" "gsm8k_tulu" "gsm8k::tulu"
    done
fi

# IFEval
if [[ ${#missing_ifeval[@]} -gt 0 ]]; then
    echo -e "\n--- Launching IFEval ---"
    for model_path in "${!missing_ifeval[@]}"; do
        launch_model_eval "$model_path" "ifeval_tulu" "ifeval::tulu"
    done
fi

# # Minerva Math
# if [[ ${#missing_minerva_math[@]} -gt 0 ]]; then
#     echo -e "\n--- Launching Minerva Math ---"
#     for model_path in "${!missing_minerva_math[@]}"; do
#         launch_model_eval "$model_path" "minerva_math_tulu" "minerva_math::tulu"
#     done
# fi

# TruthfulQA
if [[ ${#missing_truthfulqa[@]} -gt 0 ]]; then
    echo -e "\n--- Launching TruthfulQA ---"
    for model_path in "${!missing_truthfulqa[@]}"; do
        launch_model_eval "$model_path" "truthfulqa_tulu" "truthfulqa::tulu"
    done
fi

# Alpaca Eval
if [[ ${#missing_alpaca[@]} -gt 0 ]]; then
    echo -e "\n--- Launching Alpaca Eval ---"
    for model_path in "${!missing_alpaca[@]}"; do
        launch_alpaca_eval "$model_path"
    done
fi

echo -e "\nAll evaluation checks completed."