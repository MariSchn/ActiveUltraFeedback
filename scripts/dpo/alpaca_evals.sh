#!/bin/bash

# ==============================================================================
# CONFIGURATION
# ==============================================================================
THIS_DIR="${SCRATCH}/ActiveUltraFeedback"

# Default path to the evaluation script (assumes it is in the same folder as this script)
RUN_ALPACA_EVAL_SH="$SCRATCH/ActiveUltraFeedback/scripts/dpo/run_alpaca_eval.sh"

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
help_function() {
    echo "Usage: $0 --dpo_dir <path> --results_dir <path> [options]"
    echo ""
    echo "Required Options:"
    echo "  --dpo_dir <path>       Directory containing DPO model subfolders"
    echo "  --results_dir <path>   Base directory for output results"
    echo ""
    echo "Optional Options:"
    echo "  --dry_run              Print commands without executing"
    echo "  -h, --help             Show this help message"
    exit 1
}

# Initialize variables
DPO_TRAINED_DIR=""
RESULTS_BASE_DIR=""
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dpo_dir)
            DPO_TRAINED_DIR="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_BASE_DIR="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            help_function
            ;;
        *)
            echo "Unknown argument: $1"
            help_function
            ;;
    esac
done

# ==============================================================================
# VALIDATION
# ==============================================================================
missing_args=false

if [ -z "$DPO_TRAINED_DIR" ]; then
    echo "Error: --dpo_dir is required."
    missing_args=true
fi
if [ -z "$RESULTS_BASE_DIR" ]; then
    echo "Error: --results_dir is required."
    missing_args=true
fi

if [ "$missing_args" = true ]; then
    echo "----------------------------------------"
    help_function
fi

if [ ! -f "$RUN_ALPACA_EVAL_SH" ]; then
    echo "Error: Could not find run_alpaca_eval.sh at: $RUN_ALPACA_EVAL_SH"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$RESULTS_BASE_DIR"

echo "========================================"
echo "CONFIGURATION"
echo "========================================"
echo "DPO_TRAINED_DIR: $DPO_TRAINED_DIR"
echo "RESULTS_BASE:    $RESULTS_BASE_DIR"
echo "SCRIPT_TO_RUN:   $RUN_ALPACA_EVAL_SH"
echo "DRY_RUN:         $DRY_RUN"
echo "----------------------------------------"

# ==============================================================================
# MAIN LOOP
# ==============================================================================

if [ ! -d "$DPO_TRAINED_DIR" ]; then
    echo "Error: Models directory '$DPO_TRAINED_DIR' does not exist."
    exit 1
fi

count=0
skipped_count=0
total_found=0

# Iterate through subdirectories
for model_dir in "$DPO_TRAINED_DIR"/*; do
    [ -d "$model_dir" ] || continue
    ((total_found++))

    run_name="$(basename "$model_dir")"

    # Check for model weights in the root of the directory
    # We look for .safetensors, .bin, or .pt files
    has_weights=$(find "$model_dir" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \) -print -quit)
    
    if [ -z "$has_weights" ]; then
        echo "[SKIP] $run_name (no model weights found)"
        ((skipped_count++))
        continue
    fi

    # Define specific results directory for this model
    results_dir="$RESULTS_BASE_DIR/$run_name"
    log_dir="${results_dir}_logs"
    mkdir -p "$results_dir"
    mkdir -p "$log_dir"

    echo "----------------------------------------"
    echo "Processing: $run_name"
    echo "Model Path: $model_dir"
    echo "Output Dir: $results_dir"

    # Define SLURM Command
    SBATCH_CMD="sbatch --job-name=\"${run_name}_alpaca_eval\" \
    --account=a-infra01-1 \
    --output=\"${results_dir}_logs/log_%j.out\" \
    --error=\"${results_dir}_logs/log_%j.err\" \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-task=4 \
    --cpus-per-task=32 \
    --time=01:15:00 \
    --partition=normal \
    --environment=activeuf \
    --wrap=\"cd ${SCRATCH}/ActiveUltraFeedback && \
        pip install 'datasets<3.0.0' --quiet && \
        export MODEL_PATH='${model_dir}' && \
        export RESULTS_DIR='${results_dir}' && \
        export HF_HOME='${HF_HOME}' && \
        bash ./scripts/dpo/run_alpaca_eval.sh\""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would submit:"
        echo "$SBATCH_CMD"
        ((count++))
        continue
    fi

    # Execute sbatch
    echo ">>> Submitting job for $run_name..."
    eval $SBATCH_CMD
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo ">>> ERROR: sbatch submission failed for $run_name (exit code $exit_code)"
    else
        echo ">>> SUCCESS: Job submitted for $run_name"
    fi
    
    ((count++))
done

echo ""
echo "========================================="
echo "SUMMARY"
echo "========================================="
echo "Total directories found: $total_found"
echo "Jobs submitted:          $count"
echo "Models skipped:          $skipped_count"
echo "========================================="