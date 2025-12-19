#!/bin/bash

# Configuration paths (adjust as needed)
export ACCELERATE_CONFIG="./configs/accelerate/single_node.yaml"
export WANDB_DIR="${SCRATCH}/cache/wandb"
export HF_HOME="${SCRATCH}/cache/hf_cache"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

# Parse ARGS
RM_MODEL_BASE_DIR=""
DPO_MODEL_BASE_DIR=""

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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --rm_base_dir <path> --dpo_base_dir <path>"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$RM_MODEL_BASE_DIR" ]] || [[ -z "$DPO_MODEL_BASE_DIR" ]]; then
    echo "Error: Please provide all required arguments."
    echo "Usage: $0 --rm_base_dir <path> --dpo_base_dir <path>"
    exit 1
fi

echo -e "==================================="
echo -e "===== CHECKING RM EVALUATIONS ====="
echo -e "===================================\n"

# Check if RM model base directory exists
if [[ ! -d "$RM_MODEL_BASE_DIR" ]]; then
    echo "Error: RM model base directory does not exist: $RM_MODEL_BASE_DIR"
    exit 1
fi

# Get all directories in RM_MODEL_BASE_DIR
rm_dirs=()
for dir in "$RM_MODEL_BASE_DIR"/*; do
    if [[ -d "$dir" ]]; then
        rm_dirs+=("$(basename "$dir")")
    fi
done

echo "Found ${#rm_dirs[@]} directories (reward models) in $RM_MODEL_BASE_DIR"

# Check for missing RM evaluations (results.json)
echo "--- Looking for missing RM evaluations... ---"
missing_rm_evals=()
for dir_name in "${rm_dirs[@]}"; do
    results_file="$RM_MODEL_BASE_DIR/$dir_name/metrics.json"
    if [[ ! -f "$results_file" ]]; then
        echo "  Missing RM evaluation for: $dir_name (no results.json at $results_file)"
        missing_rm_evals+=("$dir_name")
    fi
done

if [[ ${#missing_rm_evals[@]} -eq 0 ]]; then
    echo "All RM evaluations are present!"
else
    echo "Found ${#missing_rm_evals[@]} missing RM evaluations"
    echo ""
    echo "--- Launching RM evaluation jobs ---"
    for dir_name in "${missing_rm_evals[@]}"; do
        rm_path="$RM_MODEL_BASE_DIR/$dir_name"

        echo "Processing: $dir_name"
        echo "  RM model path: $rm_path"
        
        # Submit RM evaluation job
        echo "  Submitting RM evaluation job..."
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
                   bash ./activeuf/reward_model/reward_bench_2.sh --model ${rm_path}
                   
                   # Update WandB run with results
                   python ./scripts/update_wandb_run.py \
                       --run_id ${dir_name} \
                       --rm_output_dir ${rm_path} \
                       --dpo_output_dir ${DPO_MODEL_BASE_DIR}/${dir_name} \
                       --project loop \
                       --entity ActiveUF
               "
        
        echo "  Job submitted for $dir_name"
        echo ""
    done
fi

echo -e "\n===================================="
echo -e "===== CHECKING DPO EVALUATIONS ====="
echo -e "====================================\n"

# Check if DPO model base directory exists
if [[ ! -d "$DPO_MODEL_BASE_DIR" ]]; then
    echo "Error: DPO model base directory does not exist: $DPO_MODEL_BASE_DIR"
    exit 1
fi

# Get all directories in DPO_MODEL_BASE_DIR
dpo_dirs=()
for dir in "$DPO_MODEL_BASE_DIR"/*; do
    if [[ -d "$dir" ]]; then
        dpo_dirs+=("$(basename "$dir")")
    fi
done

echo "Found ${#dpo_dirs[@]} directories (DPO models) in $DPO_MODEL_BASE_DIR"

# DPO benchmark files that need to exist for a eval to be considered complete
dpo_benchmark_files=(
    "results/gsm8k_tulu/metrics.json"
    "results/ifeval_tulu/metrics.json"
    "results/minerva_math_tulu/metrics.json"
    "results/truthfulqa_tulu/metrics.json"
    "results/alpaca_eval/leaderboard.csv"
)

# Declare arrays to store missing dirs per benchmark
declare -A missing_gsm8k_dirs
declare -A missing_ifeval_dirs
declare -A missing_minerva_math_dirs
declare -A missing_truthfulqa_dirs
declare -A missing_alpaca_eval_dirs

# Check for missing DPO evaluations
echo "--- Looking for missing DPO evaluations... ---"
for dir_name in "${dpo_dirs[@]}"; do
    for benchmark_file in "${dpo_benchmark_files[@]}"; do
        full_path="$DPO_MODEL_BASE_DIR/$dir_name/$benchmark_file"
        if [[ ! -f "$full_path" ]]; then
            # Store which dirs are missing each benchmark
            if [[ "$benchmark_file" == *"gsm8k"* ]]; then
                missing_gsm8k_dirs["$dir_name"]=1
            elif [[ "$benchmark_file" == *"ifeval"* ]]; then
                missing_ifeval_dirs["$dir_name"]=1
            elif [[ "$benchmark_file" == *"minerva_math"* ]]; then
                missing_minerva_math_dirs["$dir_name"]=1
            elif [[ "$benchmark_file" == *"truthfulqa"* ]]; then
                missing_truthfulqa_dirs["$dir_name"]=1
            elif [[ "$benchmark_file" == *"alpaca_eval"* ]]; then
                missing_alpaca_eval_dirs["$dir_name"]=1
            fi
        fi
    done
done

# Calculate total missing directories (unique across all benchmarks)
declare -A all_missing_dirs
for dir in "${!missing_gsm8k_dirs[@]}"; do all_missing_dirs["$dir"]=1; done
for dir in "${!missing_ifeval_dirs[@]}"; do all_missing_dirs["$dir"]=1; done
for dir in "${!missing_minerva_math_dirs[@]}"; do all_missing_dirs["$dir"]=1; done
for dir in "${!missing_truthfulqa_dirs[@]}"; do all_missing_dirs["$dir"]=1; done
for dir in "${!missing_alpaca_eval_dirs[@]}"; do all_missing_dirs["$dir"]=1; done

total_missing=${#all_missing_dirs[@]}

if [[ $total_missing -eq 0 ]]; then
    echo "All DPO evaluations are present!"
else
    echo "Found $total_missing directories with missing DPO evaluations"
fi


# Only set up and launch DPO evaluations if there are missing evaluations
if [[ $total_missing -gt 0 ]]; then
    echo "--- Launching missing DPO evaluation jobs ---"

    # Function to launch DPO evaluation for a specific benchmark
    launch_dpo_eval() {
        local model_dir=$1
        local benchmark=$2
        local task_name=$3
        local run_id=$(basename "$model_dir")
        
        echo "  Processing: $run_id"
        echo "    Benchmark: $benchmark"
        echo "    Task: $task_name"
        
        # Create results directory
        mkdir -p "${model_dir}/results/${benchmark}"
        
        # Submit DPO evaluation job
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
               
               # Update WandB run with results (will include whatever results are available)
               python ./scripts/update_wandb_run.py \
                   --run_id ${run_id} \
                   --rm_output_dir ${RM_MODEL_BASE_DIR}/${run_id} \
                   --dpo_output_dir ${model_dir} \
                   --project loop \
                   --entity ActiveUF
               "
        
        echo "    Job submitted for ${run_id}_${benchmark}"
    }

    # Launch jobs for missing GSM8K evaluations
    if [[ ${#missing_gsm8k_dirs[@]} -gt 0 ]]; then
        echo "--- Launching GSM8K evaluations (${#missing_gsm8k_dirs[@]} jobs) ---"
        for dir_name in "${!missing_gsm8k_dirs[@]}"; do
            model_path="$DPO_MODEL_BASE_DIR/$dir_name"
            launch_dpo_eval "$model_path" "gsm8k_tulu" "gsm8k::tulu"
        done
        echo ""
    fi

    # Launch jobs for missing IFEval evaluations
    if [[ ${#missing_ifeval_dirs[@]} -gt 0 ]]; then
        echo "--- Launching IFEval evaluations (${#missing_ifeval_dirs[@]} jobs) ---"
        for dir_name in "${!missing_ifeval_dirs[@]}"; do
            model_path="$DPO_MODEL_BASE_DIR/$dir_name"
            launch_dpo_eval "$model_path" "ifeval_tulu" "ifeval::tulu"
        done
        echo ""
    fi

    # Launch jobs for missing Minerva Math evaluations
    if [[ ${#missing_minerva_math_dirs[@]} -gt 0 ]]; then
        echo "--- Launching Minerva Math evaluations (${#missing_minerva_math_dirs[@]} jobs) ---"
        for dir_name in "${!missing_minerva_math_dirs[@]}"; do
            model_path="$DPO_MODEL_BASE_DIR/$dir_name"
            launch_dpo_eval "$model_path" "minerva_math_tulu" "minerva_math::tulu"
        done
        echo ""
    fi

    # Launch jobs for missing TruthfulQA evaluations
    if [[ ${#missing_truthfulqa_dirs[@]} -gt 0 ]]; then
        echo "--- Launching TruthfulQA evaluations (${#missing_truthfulqa_dirs[@]} jobs) ---"
        for dir_name in "${!missing_truthfulqa_dirs[@]}"; do
            model_path="$DPO_MODEL_BASE_DIR/$dir_name"
            launch_dpo_eval "$model_path" "truthfulqa_tulu" "truthfulqa::tulu"
        done
        echo ""
    fi

    # Launch jobs for missing Alpaca Eval evaluations
    if [[ ${#missing_alpaca_eval_dirs[@]} -gt 0 ]]; then
        echo "--- Launching Alpaca Eval evaluations (${#missing_alpaca_eval_dirs[@]} jobs) ---"
        for dir_name in "${!missing_alpaca_eval_dirs[@]}"; do
            model_path="$DPO_MODEL_BASE_DIR/$dir_name"
            run_id=$(basename "$model_path")
            results_dir="${model_path}/results/alpaca_eval"
            
            echo "  Processing: $run_id"
            echo "    Model path: $model_path"
            echo "    Results dir: $results_dir"
            
            # Create results directory
            mkdir -p "${results_dir}"
            
            # Submit Alpaca Eval job
            echo "    Submitting Alpaca Eval job..."
            sbatch --job-name="${run_id}_alpaca_eval" \
                   --account="a-infra01-1" \
                   --output="${results_dir}/log_%j.out" \
                   --error="${results_dir}/log_%j.err" \
                   --nodes=1 \
                   --ntasks=1 \
                   --gpus-per-task=4 \
                   --cpus-per-task=32 \
                   --time=00:15:00 \
                   --partition=normal \
                   --environment=activeuf \
                   --wrap="
                       cd ${SCRATCH}/ActiveUltraFeedback
                       export MODEL_PATH=\"${model_path}\"
                       export RESULTS_DIR=\"${results_dir}\"
                       export HF_HOME=\"${HF_HOME}\"
                       bash scripts/dpo/run_alpaca_eval.sh
                       
                       # Update WandB run with results (will include whatever results are available)
                       python ./scripts/update_wandb_run.py \
                           --run_id ${run_id} \
                           --rm_output_dir ${RM_MODEL_BASE_DIR}/${run_id} \
                           --dpo_output_dir ${model_path} \
                           --project loop \
                           --entity ActiveUF
                   "
            
            echo "    Job submitted for ${run_id}_alpaca_eval"
            echo ""
        done
        echo ""
    fi

    # Calculate total jobs submitted
    total_dpo_jobs=$((${#missing_gsm8k_dirs[@]} + ${#missing_ifeval_dirs[@]} + ${#missing_minerva_math_dirs[@]} + ${#missing_truthfulqa_dirs[@]} + ${#missing_alpaca_eval_dirs[@]}))
else
    total_dpo_jobs=0
    echo "No DPO evaluations to launch."
fi

echo -e "\n=========================="
echo -e "===== JOBS SUBMITTED ====="
echo -e "==========================\n"
echo "RM eval jobs submitted: ${#missing_rm_evals[@]}"
echo "GSM8K eval jobs submitted: ${#missing_gsm8k_dirs[@]}"
echo "IFEval eval jobs submitted: ${#missing_ifeval_dirs[@]}"
echo "Minerva Math eval jobs submitted: ${#missing_minerva_math_dirs[@]}"
echo "TruthfulQA eval jobs submitted: ${#missing_truthfulqa_dirs[@]}"
echo "Alpaca Eval jobs submitted: ${#missing_alpaca_eval_dirs[@]}"
