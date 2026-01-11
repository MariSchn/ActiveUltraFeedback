#!/bin/bash
# ==========================================
# Wrapper script to submit individual jobs
# Usage: ./submit_all.sh [combo_id]
#   - No args: submits all 5 jobs
#   - With arg: submits only that combo_id
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/run_tulu_single.sbatch"

# Check if sbatch script exists
if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "Error: $SBATCH_SCRIPT not found"
    exit 1
fi

submit_job() {
    local COMBO_ID=$1
    local ACQ_TYPE=$2
    local TAG=$3
    local CKPT=$4
    local BETA=$5
    local DECAY=$6
    local REPLAY=$7
    
    echo "========================================"
    echo "Submitting COMBO_ID=$COMBO_ID ($ACQ_TYPE)"
    echo "  TAG=$TAG"
    echo "  CKPT=$CKPT"
    echo "  BETA=$BETA, DECAY=$DECAY, REPLAY=$REPLAY"
    echo "========================================"
    
    sbatch \
        --export=ALL,ACQ_TYPE="$ACQ_TYPE",TAG="$TAG",CKPT="$CKPT",BETA="$BETA",DECAY="$DECAY",REPLAY="$REPLAY" \
        --job-name="${ACQ_TYPE}_run" \
        --output="./logs/looptulu/${ACQ_TYPE}_run_%j.out" \
        --error="./logs/looptulu/${ACQ_TYPE}_run_%j.err" \
        "$SBATCH_SCRIPT"
    
    echo ""
    sleep 1  # Small delay between submissions
}

# Define all configurations
# Format: submit_job COMBO_ID ACQ_TYPE TAG CKPT BETA DECAY REPLAY

run_combo_0() {
    submit_job 0 "drts" "checkpoint" \
        "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_3/actives/dpo/drts_enn_qwen_ultrafeedback_20260106-144557-020/checkpoint-checkpoint-3500" \
        "1.0" "0.999" "1000"
}

run_combo_1() {
    submit_job 1 "dts" "checkpoint" \
        "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_3/actives/dpo/dts_enn_qwen_ultrafeedback_20260106-144556-725/checkpoint-checkpoint-3500" \
        "1.0" "0.99" "1000"
}

run_combo_2() {
    submit_job 2 "deltaucb" "checkpoint" \
        "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_3/actives/dpo/deltaucb_enn_qwen_ultrafeedback_20260106-144555-800/checkpoint-checkpoint-3500" \
        "2.0" "0.999" "1000"
}

run_combo_3() {
    submit_job 3 "infomax" "checkpoint" \
        "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_3/actives/dpo/infomax_enn_qwen_ultrafeedback_20260106-144559-775/checkpoint-checkpoint-3500" \
        "1.0" "0.99" "1000"
}

run_combo_4() {
    submit_job 4 "maxminlcb" "checkpoint" \
        "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_3/actives/dpo/maxminlcb_enn_qwen_ultrafeedback_20260106-150110-615/checkpoint-checkpoint-3500" \
        "1.0" "0.99" "1000"
}

# Main logic
if [[ -n "$1" ]]; then
    # Submit specific combo
    case $1 in
        0) run_combo_0 ;;
        1) run_combo_1 ;;
        2) run_combo_2 ;;
        3) run_combo_3 ;;
        4) run_combo_4 ;;
        *) echo "Error: Invalid combo_id $1 (valid: 0-4)"; exit 1 ;;
    esac
else
    # Submit all
    echo "Submitting all 5 jobs..."
    run_combo_0
    run_combo_1
    run_combo_2
    run_combo_3
    run_combo_4
    echo "All jobs submitted!"
fi
