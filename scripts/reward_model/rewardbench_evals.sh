#!/bin/bash

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Path to your evaluation wrapper script
# Update this if the location changes
EVAL_SCRIPT_PATH="./activeuf/reward_model/reward_bench_2.sh"

# Initialize variables
MODELS_ROOT_DIR=""
SINGLE_MODEL_MODE=false
DRY_RUN=false
SUBSAMPLE_COUNT=0

# Helper function for usage
usage() {
    echo "Usage: $0 [options]"
    echo "Required:"
    echo "  --models_dir <path>      Root directory containing RM checkpoints"
    echo ""
    echo "Optional:"
    echo "  --single_model           Treat --models_dir as a specific model path"
    echo "  --subsample <int>        Run only on first N models (Default: 0/All)"
    echo "  --dry_run                Print sbatch content without submitting"
    echo "  -h, --help               Show this help"
    exit 1
}

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --models_dir)   MODELS_ROOT_DIR="$2"; shift 2 ;;
        --single_model) SINGLE_MODEL_MODE=true; shift ;;
        --subsample)    SUBSAMPLE_COUNT="$2"; shift 2 ;;
        --dry_run)      DRY_RUN=true; shift ;;
        -h|--help)      usage ;;
        *)              echo "Unknown argument: $1"; usage ;;
    esac
done

# Validation
if [ -z "$MODELS_ROOT_DIR" ]; then
    echo "Error: --models_dir is required."
    usage
fi

if [ ! -d "$MODELS_ROOT_DIR" ]; then
    echo "Error: Models directory '$MODELS_ROOT_DIR' does not exist."
    exit 1
fi

if [ ! -f "$EVAL_SCRIPT_PATH" ]; then
    echo "Warning: Eval script '$EVAL_SCRIPT_PATH' not found locally."
    echo "Proceeding, assuming it exists on the compute node relative to submission dir."
fi

echo "========================================"
echo "RM EVALUATION RUNNER"
echo "========================================"
echo "Models Dir:    $MODELS_ROOT_DIR"
echo "Eval Script:   $EVAL_SCRIPT_PATH"
echo "Mode:          $( [ "$SINGLE_MODEL_MODE" = true ] && echo "SINGLE MODEL" || echo "BATCH" )"
echo "Subsample:     $SUBSAMPLE_COUNT"
echo "Dry Run:       $DRY_RUN"
echo "========================================"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# 1. Check for valid model weights (safetensors/bin/pt)
is_valid_model_dir() {
    local dir="$1"
    local has_weights=$(find "$dir" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \) -print -quit)
    [ -n "$has_weights" ] && return 0 || return 1
}

# 2. Check if job is currently active in Slurm
job_is_active() {
    local job_name="$1"
    local job_exists=$(squeue --noheader --name="$job_name" --user="$USER")
    [ -n "$job_exists" ] && return 0 || return 1
}

# ==============================================================================
# MAIN LOGIC
# ==============================================================================

# 1. Prepare List
MODEL_LIST=()
if [ "$SINGLE_MODEL_MODE" = true ]; then
    MODEL_LIST+=("$MODELS_ROOT_DIR")
else
    shopt -s nullglob
    for path in "$MODELS_ROOT_DIR"/*; do
        [ -d "$path" ] && MODEL_LIST+=("$path")
    done
    shopt -u nullglob
fi

# 2. Subsample Logic
TOTAL_FOUND=${#MODEL_LIST[@]}
LIMIT_COUNT=$([ "$SUBSAMPLE_COUNT" -gt 0 ] && echo "$SUBSAMPLE_COUNT" || echo "$TOTAL_FOUND")
MODELS_TO_RUN=()

for ((i=0; i<LIMIT_COUNT && i<TOTAL_FOUND; i++)); do
    MODELS_TO_RUN+=("${MODEL_LIST[$i]}")
done

# 3. Processing Loop
for model_full_path in "${MODELS_TO_RUN[@]}"; do
    
    # Clean trailing slash and get name
    model_full_path="${model_full_path%/}"
    model_name=$(basename "$model_full_path")
    
    # Define Job Name
    slurm_job_name="Eval-RM-${model_name}"

    echo ""
    echo "Processing: $model_name"

    # --- VALIDATION CHECKS ---
    
    # A. Check Config
    if [ ! -f "$model_full_path/config.json" ]; then
        echo "  [SKIP] No config.json found."
        continue
    fi

    # B. Check Weights
    if ! is_valid_model_dir "$model_full_path"; then
        echo "  [SKIP] No model weights found."
        continue
    fi

    # C. Check if Completed (Output Exists)
    # TODO: Adjust "metrics.json" to whatever your reward_bench_2.sh outputs!
    if [ -f "$model_full_path/metrics.json" ]; then 
        echo "  [SKIP] Results already exist in model dir."
        continue
    fi

    # D. Check if Active (Slurm)
    if job_is_active "$slurm_job_name"; then
        echo "  [SKIP] Job '$slurm_job_name' is currently active."
        continue
    fi

    echo "  [SUBMIT] Submitting evaluation job..."

    # --- GENERATE SBATCH ---
    sbatch_content=$(cat << EOF
#!/bin/bash
#SBATCH --job-name=${slurm_job_name}
#SBATCH --output=${model_full_path}/eval_%x.%j.out
#SBATCH --error=${model_full_path}/eval_%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --environment=activeuf_dev
#SBATCH --account=a-infra01-1
#SBATCH --exclude=nid[006845,006851,006854-006855,006752,006755,006787-006788,007095-007096,007098-007099,006824,006826,006845,006851,006854-006855,006752,006755,006787-006788,007095-007096,007098-007099,006708,006789-006791,007096,007098-007099,007104,006869,006875,007171,007173-007174,007182,007184,007201,007203,007189-007190,007234-007235,007087]

# Echo info for logs
echo "Starting RM Evaluation"
echo "Evaluating RM at: $model_full_path"

# Run the wrapper script
# We assume the script handles the python call internally
bash $EVAL_SCRIPT_PATH --model "$model_full_path"

echo "Evaluation finished."
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