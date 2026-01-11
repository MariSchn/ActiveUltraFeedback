#!/bin/bash

# Configuration paths (adjust as needed)
export ACCELERATE_CONFIG="./configs/accelerate/single_node.yaml"
export XDG_CACHE_HOME="${SCRATCH}/cache"
export WANDB_DIR="${SCRATCH}/cache/wandb"
export WANDB_CACHE_DIR="${SCRATCH}/cache/wandb"
export HF_HOME="${SCRATCH}/huggingface"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export TRITON_CACHE_DIR="${SCRATCH}/cache/triton"
export VLLM_NO_USAGE_STATS=1  # Prevents vLLM from writing usage stats to home

# Global Benchmark Definitions
# Note: Minerva Math is commented out but preserved for future use
BENCHMARK_FILES=(
    "results/gsm8k_tulu/metrics.json"
    "results/ifeval_tulu/metrics.json"
    # "results/minerva_math_tulu/metrics.json"
    "results/truthfulqa_tulu/metrics.json"
    "results/alpaca_eval/activeuf/leaderboard.csv"
)

# Parse ARGS
RM_MODEL_BASE_DIR=""
DPO_MODEL_BASE_DIR=""
IPO_MODEL_BASE_DIR=""
SIMPO_MODEL_BASE_DIR=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --rm_base_dir)
            RM_MODEL_BASE_DIR="$2"
            shift 2
            ;;
        --dpo_base_dir)
            DPO_MODEL_BASE_DIR="$2"
            shift 2
            ;;
        --ipo_base_dir)
            IPO_MODEL_BASE_DIR="$2"
            shift 2
            ;;
        --simpo_base_dir)
            SIMPO_MODEL_BASE_DIR="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --rm_base_dir <path> --dpo_base_dir <path> [--ipo_base_dir <path>] [--simpo_base_dir <path>] [--dry_run]"
            exit 1
            ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "!!! DRY RUN MODE ENABLED: No jobs will be submitted !!!"
fi

# Validate required arguments
if [[ -z "$RM_MODEL_BASE_DIR" ]] || [[ -z "$DPO_MODEL_BASE_DIR" ]]; then
    echo "Error: Please provide at least RM and DPO base directories."
    echo "Usage: $0 --rm_base_dir <path> --dpo_base_dir <path> ..."
    exit 1
fi

# ==============================================================================
# Helper Function: Launch Generic Evaluation Job (Shared for DPO, IPO, SimPO)
# ==============================================================================
launch_model_eval() {
    local model_dir=$1
    local benchmark=$2
    local task_name=$3
    local run_id=$(basename "$model_dir")
    local model_type_label=$4 # "DPO", "IPO", or "SIMPO"

    echo "  Processing [$model_type_label]: $run_id"
    echo "    Benchmark: $benchmark"
    echo "    Task: $task_name"
    
    # Construct WandB Update Arguments based on model type
    WANDB_UPDATE_ARGS="--run_id ${run_id} \
                       --rm_output_dir ${RM_MODEL_BASE_DIR}/${run_id} \
                       --dpo_output_dir ${DPO_MODEL_BASE_DIR}/${run_id} \
                       --project loop \
                       --entity ActiveUF"

    if [[ "$model_type_label" == "IPO" ]]; then
        WANDB_UPDATE_ARGS="${WANDB_UPDATE_ARGS} --ipo_output_dir ${model_dir}"
    elif [[ "$model_type_label" == "SIMPO" ]]; then
        WANDB_UPDATE_ARGS="${WANDB_UPDATE_ARGS} --simpo_output_dir ${model_dir}"
    fi
    
    # Create results directory
    if [ "$DRY_RUN" = false ]; then
        mkdir -p "${model_dir}/results/${benchmark}"
    fi
    
    # Submit Evaluation job
    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY RUN] Would submit sbatch job for: ${run_id}_${benchmark}"
    else
        echo "    Submitting evaluation job..."
        sbatch --job-name="${run_id}_${benchmark}" \
               --account="a-infra01-1" \
               --output="${model_dir}/results/${benchmark}/eval_%j.log" \
               --nodes=1 \
               --ntasks=1 \
               --gpus-per-task=4 \
               --time=5:00:00 \
               --partition=normal \
               --wrap="
                export VLLM_WORKER_MULTIPROC_METHOD=spawn
                export PROJECT_ROOT_AT=$SCRATCH/projects/ActiveUltraFeedback/resources/olmes
                export PROJECT_NAME=olmes
                export PACKAGE_NAME=oe_eval
                export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
                export WANDB_API_KEY_FILE_AT=$HOME/.wandb-api-key
                export HF_HOME=$SCRATCH/cache/hf_cache
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
\$WANDB_API_KEY_FILE_AT,\
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
                
                srun --nodes=1 --ntasks=1 --gpus-per-task=4 \$CONTAINER_ARGS bash -c \"exec python3 -m oe_eval.launch --task=${task_name} --output-dir=${model_dir}/results/${benchmark} \$EVAL_ARGS\"
                
                # Update WandB
                python ./scripts/update_wandb_run.py ${WANDB_UPDATE_ARGS}
                "
        echo "    Job submitted for ${run_id}_${benchmark}"
    fi
}

launch_alpaca_eval() {
    local model_dir=$1
    local run_id=$(basename "$model_dir")
    local model_type_label=$2

    local results_dir="${model_dir}/results/alpaca_eval"
    
    echo "  Processing [$model_type_label]: $run_id (Alpaca Eval)"
    
    # Construct WandB Update Arguments
    WANDB_UPDATE_ARGS="--run_id ${run_id} \
                       --rm_output_dir ${RM_MODEL_BASE_DIR}/${run_id} \
                       --dpo_output_dir ${DPO_MODEL_BASE_DIR}/${run_id} \
                       --project loop \
                       --entity ActiveUF"

    if [[ "$model_type_label" == "IPO" ]]; then
        WANDB_UPDATE_ARGS="${WANDB_UPDATE_ARGS} --ipo_output_dir ${model_dir}"
    elif [[ "$model_type_label" == "SIMPO" ]]; then
        WANDB_UPDATE_ARGS="${WANDB_UPDATE_ARGS} --simpo_output_dir ${model_dir}"
    fi
    
    if [ "$DRY_RUN" = false ]; then
        mkdir -p "${results_dir}"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY RUN] Would submit sbatch job for: ${run_id}_alpaca_eval"
    else
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
                   cd ${SCRATCH}/ActiveUltraFeedback
                   pip install \"datasets<3.0.0\" --quiet
                   export MODEL_PATH=\"${model_dir}\"
                   export RESULTS_DIR=\"${results_dir}\"
                   export HF_HOME=\"${HF_HOME}\"
                   export XDG_CACHE_HOME=\"${SCRATCH}/cache\"
                   export WANDB_DIR=\"${SCRATCH}/cache/wandb\"
                   export WANDB_CACHE_DIR=\"${SCRATCH}/cache/wandb\"
                   export TRITON_CACHE_DIR=\"${SCRATCH}/cache/triton\"
                   export VLLM_DISABLE_COMPILE_CACHE=1
                   bash scripts/dpo/run_alpaca_eval.sh
                   
                   # Update WandB
                   python ./scripts/update_wandb_run.py ${WANDB_UPDATE_ARGS}
               "
        echo "    Job submitted for ${run_id}_alpaca_eval"
    fi
}

# ==============================================================================
# Helper Function: Process Directory Evaluations (Generic for DPO, IPO, SimPO)
# ==============================================================================
process_eval_checks() {
    local base_dir=$1
    local type_label=$2

    echo -e "\n===================================="
    echo -e "===== CHECKING $type_label EVALUATIONS ====="
    echo -e "====================================\n"

    if [[ ! -d "$base_dir" ]]; then
        echo "Warning: $type_label base directory does not exist or was not provided: $base_dir"
        return
    fi

    # Get all directories
    local model_dirs=()
    for dir in "$base_dir"/*; do
        if [[ -d "$dir" ]]; then
            model_dirs+=("$(basename "$dir")")
        fi
    done

    echo "Found ${#model_dirs[@]} directories ($type_label models) in $base_dir"

    # Tracking missing evals
    declare -A missing_gsm8k
    declare -A missing_ifeval
    # declare -A missing_minerva_math
    declare -A missing_truthfulqa
    declare -A missing_alpaca

    echo "--- Looking for missing $type_label evaluations... ---"
    for dir_name in "${model_dirs[@]}"; do
        for benchmark_file in "${BENCHMARK_FILES[@]}"; do
            full_path="$base_dir/$dir_name/$benchmark_file"
            if [[ ! -f "$full_path" ]]; then
                if [[ "$benchmark_file" == *"gsm8k"* ]]; then
                    missing_gsm8k["$dir_name"]=1
                elif [[ "$benchmark_file" == *"ifeval"* ]]; then
                    missing_ifeval["$dir_name"]=1
                # elif [[ "$benchmark_file" == *"minerva_math"* ]]; then
                #     missing_minerva_math["$dir_name"]=1
                elif [[ "$benchmark_file" == *"truthfulqa"* ]]; then
                    missing_truthfulqa["$dir_name"]=1
                elif [[ "$benchmark_file" == *"alpaca_eval"* ]]; then
                    missing_alpaca["$dir_name"]=1
                fi
            fi
        done
    done

    # Launch Jobs
    # GSM8K
    if [[ ${#missing_gsm8k[@]} -gt 0 ]]; then
        echo "--- Launching GSM8K ($type_label) ---"
        for dir_name in "${!missing_gsm8k[@]}"; do
            launch_model_eval "$base_dir/$dir_name" "gsm8k_tulu" "gsm8k::tulu" "$type_label"
        done
    fi

    # IFEval
    if [[ ${#missing_ifeval[@]} -gt 0 ]]; then
        echo "--- Launching IFEval ($type_label) ---"
        for dir_name in "${!missing_ifeval[@]}"; do
            launch_model_eval "$base_dir/$dir_name" "ifeval_tulu" "ifeval::tulu" "$type_label"
        done
    fi

    # # Minerva Math
    # if [[ ${#missing_minerva_math[@]} -gt 0 ]]; then
    #     echo "--- Launching Minerva Math ($type_label) ---"
    #     for dir_name in "${!missing_minerva_math[@]}"; do
    #         launch_model_eval "$base_dir/$dir_name" "minerva_math_tulu" "minerva_math::tulu" "$type_label"
    #     done
    # fi

    # TruthfulQA
    if [[ ${#missing_truthfulqa[@]} -gt 0 ]]; then
        echo "--- Launching TruthfulQA ($type_label) ---"
        for dir_name in "${!missing_truthfulqa[@]}"; do
            launch_model_eval "$base_dir/$dir_name" "truthfulqa_tulu" "truthfulqa::tulu" "$type_label"
        done
    fi

    # Alpaca Eval
    if [[ ${#missing_alpaca[@]} -gt 0 ]]; then
        echo "--- Launching Alpaca Eval ($type_label) ---"
        for dir_name in "${!missing_alpaca[@]}"; do
            launch_alpaca_eval "$base_dir/$dir_name" "$type_label"
        done
    fi
}


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

# 1. Process RM Evaluations (Completely Separate, as requested)
echo -e "==================================="
echo -e "===== CHECKING RM EVALUATIONS ====="
echo -e "===================================\n"

if [[ ! -d "$RM_MODEL_BASE_DIR" ]]; then
    echo "Error: RM model base directory does not exist: $RM_MODEL_BASE_DIR"
    exit 1
fi

rm_dirs=()
for dir in "$RM_MODEL_BASE_DIR"/*; do
    if [[ -d "$dir" ]]; then
        rm_dirs+=("$(basename "$dir")")
    fi
done
echo "Found ${#rm_dirs[@]} directories (reward models) in $RM_MODEL_BASE_DIR"

missing_rm_evals=()
for dir_name in "${rm_dirs[@]}"; do
    results_file="$RM_MODEL_BASE_DIR/$dir_name/metrics.json"
    if [[ ! -f "$results_file" ]]; then
        echo "  Missing RM evaluation for: $dir_name"
        missing_rm_evals+=("$dir_name")
    fi
done

if [[ ${#missing_rm_evals[@]} -gt 0 ]]; then
    echo "--- Launching RM evaluation jobs ---"
    for dir_name in "${missing_rm_evals[@]}"; do
        rm_path="$RM_MODEL_BASE_DIR/$dir_name"
        echo "Processing: $dir_name"
        
        # NOTE: RM evals are kept separate from IPO/SimPO logic.
        # This job updates WandB with RM + DPO scores only.
        
        if [ "$DRY_RUN" = true ]; then
            echo "    [DRY RUN] Would submit sbatch job for: rm_eval_${dir_name}"
        else
            sbatch --job-name="rm_eval_${dir_name}" \
                   --account="a-infra01-1" \
                   --output="${rm_path}/eval_%j.log" \
                   --nodes=1 \
                   --ntasks=1 \
                   --gpus-per-task=4 \
                   --time=4:00:00 \
                   --partition=normal \
                   --environment=activeuf_dev \
                   --wrap="
                       export VLLM_DISABLE_COMPILE_CACHE=1
                       bash ./activeuf/reward_model/reward_bench_2.sh --model ${rm_path}
                       
                       # Update WandB run
                       python ./scripts/update_wandb_run.py \
                           --run_id ${dir_name} \
                           --rm_output_dir ${rm_path} \
                           --dpo_output_dir ${DPO_MODEL_BASE_DIR}/${dir_name} \
                           --project loop \
                           --entity ActiveUF
                   "
            echo "  Job submitted for $dir_name"
        fi
    done
else
    echo "All RM evaluations are present!"
fi


# 2. Process DPO Evaluations
process_eval_checks "$DPO_MODEL_BASE_DIR" "DPO"

# 3. Process IPO Evaluations (Optional)
if [[ -n "$IPO_MODEL_BASE_DIR" ]]; then
    process_eval_checks "$IPO_MODEL_BASE_DIR" "IPO"
fi

# 4. Process SimPO Evaluations (Optional)
if [[ -n "$SIMPO_MODEL_BASE_DIR" ]]; then
    process_eval_checks "$SIMPO_MODEL_BASE_DIR" "SimPO"
fi

echo -e "\nAll evaluation checks completed."