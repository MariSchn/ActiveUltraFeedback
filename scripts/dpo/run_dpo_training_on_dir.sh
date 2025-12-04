#!/bin/bash

# ================= INPUT & VALIDATION =================
# Usage: 
#   Standard: ./run_dpo_jobs.sh <DATASETS_ROOT_DIR> <BASE_OUTPUT_DIR> [SEED]
#   Single:   ./run_dpo_jobs.sh --single_dataset <TARGET_DATASET_PATH> <BASE_OUTPUT_DIR> [SEED]

SINGLE_DATASET_MODE=false
POSITIONAL_ARGS=()

# Parse arguments to find optional flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --single_dataset)
      SINGLE_DATASET_MODE=true
      shift # Remove --single_dataset from processing
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # Save positional args
      shift
      ;;
  esac
done

# Restore positional arguments for easier handling
set -- "${POSITIONAL_ARGS[@]}"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 [--single_dataset] <input_path> <base_output_dir> [seed]"
    exit 1
fi

INPUT_PATH="$1"
BASE_OUTPUT_DIR="$2"
# Use 3rd argument as seed, default to 42 if not provided
SEED="${3:-42}"

# Check if input directory exists
if [ ! -d "$INPUT_PATH" ]; then
    echo "Error: Input path '$INPUT_PATH' does not exist."
    exit 1
fi

# Create output dir if it doesn't exist
mkdir -p "$BASE_OUTPUT_DIR"

echo "========================================"
if [ "$SINGLE_DATASET_MODE" = true ]; then
    echo "Mode:          SINGLE DATASET"
    echo "Target Path:   $INPUT_PATH"
else
    echo "Mode:          BATCH (Iterate subfolders)"
    echo "Root Dir:      $INPUT_PATH"
fi
echo "Output Dir:    $BASE_OUTPUT_DIR"
echo "Seed:          $SEED"
echo "========================================"

# Prepare the list of datasets to process based on the flag
DATASET_LIST=()

if [ "$SINGLE_DATASET_MODE" = true ]; then
    # Case A: Only process the specific folder provided
    DATASET_LIST+=("$INPUT_PATH")
else
    # Case B: Iterate through all subfolders in the root dir
    # Using nullglob ensures the loop doesn't run if no files match
    shopt -s nullglob
    for path in "$INPUT_PATH"/*; do
        DATASET_LIST+=("$path")
    done
    shopt -u nullglob
fi

# ------------------------------------------------------------------
# HELPER: ROBUST JOB CHECKER
# ------------------------------------------------------------------
job_is_active() {
    local job_name="$1"
    # Queries squeue for the current user and looks for exact job name match
    # --noheader: removes column headers
    # --name: matches specific job name provided in SBATCH
    # --user: ensures we only check our own jobs
    local job_exists=$(squeue --noheader --name="$job_name" --user="$USER")

    if [ -n "$job_exists" ]; then
        return 0 # True (Active)
    fi
    return 1 # False
}

# Iterate over the prepared list
for dataset_path in "${DATASET_LIST[@]}"; do
    if [ -d "$dataset_path" ]; then
        
        # Remove trailing slash if present for cleaner basename
        dataset_path="${dataset_path%/}"
        
        # Extract the folder name (e.g., 'qwen_3_235b' or 'qwen_3_235b_1000')
        dataset_name=$(basename "$dataset_path")
        
        # Create a Unique Job Name for Slurm
        slurm_job_name="DPO-${dataset_name}"

        # 1. ROBUSTNESS CHECK: FILES (Completed)
        # Look for any folder in the output dir that ends with "-[dataset_name]"
        if find "$BASE_OUTPUT_DIR" -maxdepth 1 -type d -name "*-$dataset_name" 2>/dev/null | grep -q .; then
            echo "[SKIP] Output for '$dataset_name' already exists. Skipping..."
            continue
        fi

        # 2. ROBUSTNESS CHECK: SLURM (Ongoing)
        if job_is_active "$slurm_job_name"; then
            echo "[SKIP] Job '$slurm_job_name' is currently active/pending in Slurm. Skipping..."
            continue
        fi

        echo "[SUBMIT] Submitting job for '$dataset_name'..."

        # 3. DYNAMIC SBATCH GENERATION
        # Variables without slash (e.g. $dataset_path) are expanded NOW (by this script).
        # Variables with slash (e.g. \$SLURM_JOB_ID) are expanded LATER (by the compute node).
        
        sbatch << EOF
#!/bin/bash
#SBATCH -A a-infra01-1
#SBATCH --job-name=${slurm_job_name}
#SBATCH --output=logs/dpo/O-%x.%j
#SBATCH --error=logs/dpo/E-%x.%j
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=05:00:00

# --- Environment Setup on Compute Node ---
BASE_CACHE_DIR="\$SCRATCH"
export HF_HOME=\$BASE_CACHE_DIR/hf_home
export VLLM_CACHE_DIR=\$BASE_CACHE_DIR/vllm_cache
export WANDB_PROJECT=DPO
export ACCELERATE_DIR="\${ACCELERATE_DIR:-/accelerate}"

# Create logs dir if not exists
mkdir -p logs/dpo

# --- Network Setup for Multi-Node ---
MAIN_PROCESS_IP=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
MAIN_PROCESS_PORT=29500
NUM_PROCESSES=\$(expr \$SLURM_NNODES \* \$SLURM_GPUS_ON_NODE)

# --- Construct the Command ---
# $dataset_path, $BASE_OUTPUT_DIR, and $SEED are injected from the outer script here.

CMD="accelerate launch \\
    --config_file=\$SCRATCH/ActiveUltraFeedback/configs/accelerate/deepspeed2.yaml \\
    --num_processes \$NUM_PROCESSES \\
    --num_machines \$SLURM_NNODES \\
    --machine_rank \\\$SLURM_NODEID \\
    --main_process_ip \$MAIN_PROCESS_IP \\
    --main_process_port \$MAIN_PROCESS_PORT \\
    -m activeuf.dpo.training \\
    --config_path \$SCRATCH/ActiveUltraFeedback/configs/dpo_training.yaml \\
    --slurm_job_id \$SLURM_JOB_ID \\
    --dataset_path $dataset_path \\
    --base_output_dir $BASE_OUTPUT_DIR \\
    --beta 0.1 \\
    --learning_rate 2e-5 \\
    --seed $SEED \\
    --num_epochs 3"

echo "\$CMD"

START=\$(date +%s)

srun --environment=activeuf_dev bash -c "\$CMD"

END=\$(date +%s)
DURATION=\$(( END - START ))

echo "Job ended at: \$(date)"
echo "Total execution time: \$DURATION seconds"
EOF

    fi
done