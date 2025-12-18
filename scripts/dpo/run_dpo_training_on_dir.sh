#!/bin/bash

# ================= INPUT & VALIDATION =================
# Usage: 
#   Standard: ./run_dpo_jobs.sh <DATASETS_ROOT_DIR> <BASE_OUTPUT_DIR> [SEED]
#   Single:   ./run_dpo_jobs.sh --single_dataset <TARGET_DATASET_PATH> <BASE_OUTPUT_DIR> [SEED]

TEMPLATE_FILE="activeuf/cpo/training.sbatch"

SINGLE_DATASET_MODE=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --single_dataset)
      SINGLE_DATASET_MODE=true
      shift 
      ;;
    *)
      POSITIONAL_ARGS+=("$1") 
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 [--single_dataset] <input_path> <base_output_dir> [seed]"
    exit 1
fi

INPUT_PATH="$1"
BASE_OUTPUT_DIR="$2"
SEED="${3:-42}"

if [ ! -d "$INPUT_PATH" ]; then
    echo "Error: Input path '$INPUT_PATH' does not exist."
    exit 1
fi

if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: Template file '$TEMPLATE_FILE' not found."
    exit 1
fi

# Ensure directories exist
mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "logs/dpo"

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
echo "Template:      $TEMPLATE_FILE"
echo "========================================"

DATASET_LIST=()
if [ "$SINGLE_DATASET_MODE" = true ]; then
    DATASET_LIST+=("$INPUT_PATH")
else
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
    local job_exists=$(squeue --noheader --name="$job_name" --user="$USER")
    if [ -n "$job_exists" ]; then
        return 0 
    fi
    return 1 
}

# Iterate over the prepared list
for dataset_path in "${DATASET_LIST[@]}"; do
    if [ -d "$dataset_path" ]; then
        
        # Clean path strings
        dataset_path="${dataset_path%/}"
        dataset_name=$(basename "$dataset_path")
        slurm_job_name="DPO-${dataset_name}"

        # 1. ROBUSTNESS CHECK: FILES
        if find "$BASE_OUTPUT_DIR" -maxdepth 1 -type d -name "*-$dataset_name" 2>/dev/null | grep -q .; then
            echo "[SKIP] Output for '$dataset_name' already exists."
            continue
        fi

        # 2. ROBUSTNESS CHECK: SLURM
        if job_is_active "$slurm_job_name"; then
            echo "[SKIP] Job '$slurm_job_name' is active."
            continue
        fi

        echo "[SUBMIT] Submitting job for '$dataset_name'..."

        # 3. EXPORT SUBMISSION (CPO Style)
        # We export variables so the sbatch script can read them via $MyVar
        # ALL -> keeps current env vars
        # Then we override specific ones
        
        sbatch \
            --job-name="$slurm_job_name" \
            --export=ALL,MyDatasetPath="$dataset_path",MyOutputDir="$BASE_OUTPUT_DIR",MySeed="$SEED" \
            "$TEMPLATE_FILE"

        sleep 0.5
    fi
done