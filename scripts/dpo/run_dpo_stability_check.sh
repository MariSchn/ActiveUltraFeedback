#!/bin/bash

# ================= DPO STABILITY CHECKER =================
# Launches 5 DPO training jobs with different seeds for stability analysis.
#
# Usage:
#   ./run_dpo_stability_check.sh <DATASET_PATH> [BASE_OUTPUT_DIR]
#
# Example:
#   ./run_dpo_stability_check.sh /path/to/dataset
#   ./run_dpo_stability_check.sh /path/to/dataset /custom/output/dir
#
# This will submit 5 jobs with seeds: 42, 3848573921, 249857201, 3985729302, 2837465910
# Output directories will be named with the seed suffix for easy comparison.
# =========================================================

# Predefined seeds for stability checking
# SEEDS=(42 3848573921 249857201 3985729302 2837465910)
SEEDS=(42 42 42 42)

TEMPLATE_FILE="activeuf/dpo/training.sbatch"

# ================= INPUT VALIDATION =================
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <dataset_path> [base_output_dir]"
    echo ""
    echo "Arguments:"
    echo "  dataset_path     Path to the dataset directory"
    echo "  base_output_dir  (Optional) Base output directory for models"
    echo "                   Default: \$SCRATCH/models/dpo_stability"
    echo ""
    echo "This script launches 5 DPO jobs with seeds: ${SEEDS[*]}"
    exit 1
fi

DATASET_PATH="${1%/}"  # Remove trailing slash if present
BASE_OUTPUT_DIR="${2:-$SCRATCH/models/dpo_stability}"

# Note: We don't check if DATASET_PATH exists because it can be a remote HuggingFace dataset

if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: Template file '$TEMPLATE_FILE' not found."
    echo "Make sure you're running from the ActiveUltraFeedback directory."
    exit 1
fi

# Extract dataset name for job naming
DATASET_NAME=$(basename "$DATASET_PATH")

# Ensure directories exist
mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "logs/dposeeds"

echo "========================================"
echo "DPO STABILITY CHECK"
echo "========================================"
echo "Dataset:       $DATASET_PATH"
echo "Dataset Name:  $DATASET_NAME"
echo "Output Dir:    $BASE_OUTPUT_DIR"
echo "Template:      $TEMPLATE_FILE"
echo "Seeds:         ${SEEDS[*]}"
echo "========================================"
echo ""

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

# Track submitted jobs
SUBMITTED_COUNT=0
SKIPPED_COUNT=0

# Launch jobs for each seed
for SEED in "${SEEDS[@]}"; do
    # Create seed-specific output directory
    # SEED_OUTPUT_DIR="${BASE_OUTPUT_DIR}/seed_${SEED}"
    SLURM_JOB_NAME="DPO-${DATASET_NAME}-s${SEED}"
    
    echo "[SEED $SEED] Processing..."
    
    # # 1. Check if output already exists
    # if find "$SEED_OUTPUT_DIR" -maxdepth 1 -type d -name "*-$DATASET_NAME" 2>/dev/null | grep -q .; then
    #     echo "  [SKIP] Output for seed $SEED already exists in $SEED_OUTPUT_DIR"
    #     ((SKIPPED_COUNT++))
    #     continue
    # fi
    
    # # 2. Check if job is already running
    # if job_is_active "$SLURM_JOB_NAME"; then
    #     echo "  [SKIP] Job '$SLURM_JOB_NAME' is already active."
    #     ((SKIPPED_COUNT++))
    #     continue
    # fi
    
    # 3. Create seed-specific output directory
    mkdir -p "$BASE_OUTPUT_DIR"
    
    # 4. Submit the job
    echo "  [SUBMIT] Submitting job '$SLURM_JOB_NAME'..."
    
    sbatch \
        --job-name="$SLURM_JOB_NAME" \
        --export=ALL,MyDatasetPath="$DATASET_PATH",MyOutputDir="$BASE_OUTPUT_DIR",MySeed="$SEED" \
        "$TEMPLATE_FILE"
    
    ((SUBMITTED_COUNT++))
    sleep 0.5
done

echo ""
echo "========================================"
echo "STABILITY CHECK SUMMARY"
echo "========================================"
echo "Total seeds:   ${#SEEDS[@]}"
echo "Submitted:     $SUBMITTED_COUNT"
echo "Skipped:       $SKIPPED_COUNT"
echo "========================================"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Results will be in: $BASE_OUTPUT_DIR/seed_*/"
