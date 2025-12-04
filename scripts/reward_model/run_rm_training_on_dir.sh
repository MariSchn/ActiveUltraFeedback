#!/bin/bash

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- FIXED CONFIGURATION ---
FIXED_CONFIG_PATH="$SCRATCH/ActiveUltraFeedback/configs/rm_training.yaml"

# Initialize variables
DATASET_INPUT_PATH=""
BASE_OUTPUT_DIR=""
SEED="${SEED:-42}"
SINGLE_DATASET_MODE=false
DRY_RUN=false

# Helper function for usage
usage() {
    echo "Usage: $0 [options]"
    echo "Required:"
    echo "  --dataset_path <path>    Path to dataset root dir (or single dataset with flag)"
    echo "  --output_dir <path>      Base directory for saving models"
    echo ""
    echo "Optional:"
    echo "  --single_dataset         Treat --dataset_path as a specific dataset, not a root dir"
    echo "  --seed <int>             Random seed (default: 42)"
    echo "  --dry_run                Print sbatch content without submitting"
    echo "  -h, --help               Show this help"
    exit 1
}

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_path) DATASET_INPUT_PATH="$2"; shift 2 ;;
        --output_dir)   BASE_OUTPUT_DIR="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --single_dataset) SINGLE_DATASET_MODE=true; shift ;;
        --dry_run)      DRY_RUN=true; shift ;;
        -h|--help)      usage ;;
        *)              echo "Unknown argument: $1"; usage ;;
    esac
done

# Validation
missing_args=false
[ -z "$DATASET_INPUT_PATH" ] && echo "Error: --dataset_path is required." && missing_args=true
[ -z "$BASE_OUTPUT_DIR" ] && echo "Error: --output_dir is required." && missing_args=true

if [ "$missing_args" = true ]; then usage; fi

# Check inputs exist
if [ ! -d "$DATASET_INPUT_PATH" ]; then
    echo "Error: Dataset path '$DATASET_INPUT_PATH' does not exist."
    exit 1
fi

# Create output dir
mkdir -p "$BASE_OUTPUT_DIR"

echo "========================================"
echo "RM TRAINING RUNNER"
echo "========================================"
echo "Mode:          $( [ "$SINGLE_DATASET_MODE" = true ] && echo "SINGLE DATASET" || echo "BATCH (Iterate Subfolders)" )"
echo "Input Path:    $DATASET_INPUT_PATH"
echo "Output Base:   $BASE_OUTPUT_DIR"
echo "Config (Fixed):$FIXED_CONFIG_PATH"
echo "Seed:          $SEED"
echo "Dry Run:       $DRY_RUN"
echo "========================================"

# ==============================================================================
# DATASET LIST PREPARATION
# ==============================================================================

DATASET_LIST=()

if [ "$SINGLE_DATASET_MODE" = true ]; then
    DATASET_LIST+=("$DATASET_INPUT_PATH")
else
    shopt -s nullglob
    for path in "$DATASET_INPUT_PATH"/*; do
        if [ -d "$path" ]; then
            DATASET_LIST+=("$path")
        fi
    done
    shopt -u nullglob
fi

# ==============================================================================
# PROCESSING LOOP
# ==============================================================================

# ------------------------------------------------------------------
# CORRECTED FUNCTION: ROBUST JOB CHECK
# ------------------------------------------------------------------
job_is_active() {
    local job_name="$1"
    
    # We ask Slurm specifically: "Do you have any jobs with THIS name for THIS user?"
    # --noheader: Don't print the column titles
    # --name: Exact name match (handles spacing correctly)
    # --user: Ensure we only check our own jobs
    
    local job_exists=$(squeue --noheader --name="$job_name" --user="$USER")

    if [ -n "$job_exists" ]; then
        return 0 # True (Active)
    fi
    return 1 # False
}
# ------------------------------------------------------------------

for dataset_full_path in "${DATASET_LIST[@]}"; do
    
    # Clean trailing slash
    dataset_full_path="${dataset_full_path%/}"
    dataset_name=$(basename "$dataset_full_path")
    
    # Define specific output directory for this model
    job_output_dir="${BASE_OUTPUT_DIR}/${dataset_name}"
    
    # Define unique job name for Slurm
    slurm_job_name="RM-${dataset_name}"

    echo ""
    echo "Processing: $dataset_name"

    # 1. CHECK: Output Exists (Completed)
    if [ -f "$job_output_dir/config.json" ]; then
        echo "  [SKIP] Output already exists at $job_output_dir"
        continue
    fi

    # 2. CHECK: Job Running (Ongoing) - NOW ROBUST
    if job_is_active "$slurm_job_name"; then
        echo "  [SKIP] Job '$slurm_job_name' is currently active in Slurm."
        continue
    fi

    echo "  [SUBMIT] Submitting job..."

    # 3. GENERATE SBATCH
    sbatch_content=$(cat << EOF
#!/bin/bash
#SBATCH --job-name=${slurm_job_name}
#SBATCH -D .
#SBATCH -A a-infra01-1
#SBATCH --output=logs/rm/O-%x.%j
#SBATCH --error=logs/rm/E-%x.%j
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=02:30:00
#SBATCH --environment=activeuf_dev

# --- Env Setup ---
export GPUS_PER_NODE=4
export HF_HOME=\$SCRATCH/huggingface
export WANDB_PROJECT="RM-Training"

# Ensure logs dir exists
mkdir -p logs/rm

# --- Network Setup ---
head_node_ip=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)

# --- Paths ---
REPO_ROOT="\$SCRATCH/ActiveUltraFeedback"
ACCELERATE_CONFIG="\$REPO_ROOT/configs/accelerate/deepspeed2.yaml"
PYTHON_SCRIPT="\$REPO_ROOT/activeuf/reward_model/training.py"

export ACCELERATE_DIR="\${ACCELERATE_DIR:-/accelerate}"

# --- Launch Command ---
LAUNCHER="accelerate launch \\
    --config_file \$ACCELERATE_CONFIG \\
    --num_processes \$((SLURM_NNODES * GPUS_PER_NODE)) \\
    --num_machines \$SLURM_NNODES \\
    --rdzv_backend c10d \\
    --main_process_ip \$head_node_ip \\
    --main_process_port 29500"

SCRIPT_ARGS=" \\
    --output_dir ${job_output_dir} \\
    --reward_config ${FIXED_CONFIG_PATH} \\
    --dataset_path ${dataset_full_path} \\
    --seed ${SEED}"

CMD="\$LAUNCHER \$PYTHON_SCRIPT \$SCRIPT_ARGS"

echo "Command: \$CMD"

START=\$(date +%s)

cd \$REPO_ROOT

srun \$CMD

END=\$(date +%s)
DURATION=\$(( END - START ))

echo "Job ended at: \$(date)"
echo "Total execution time: \$DURATION seconds"
EOF
)

    if [ "$DRY_RUN" = true ]; then
        echo "---------------- DRY RUN: SBATCH CONTENT ----------------"
        echo "$sbatch_content"
        echo "---------------------------------------------------------"
    else
        echo "$sbatch_content" | sbatch
    fi

done