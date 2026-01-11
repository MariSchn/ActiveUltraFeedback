#!/usr/bin/env bash
set -euo pipefail

# Script to organize all sweep datasets by acquisition function
# Based on the sweep IDs from the user's notes

SCRATCH="${SCRATCH:-/iopsstor/scratch/cscs/dmelikidze}"
SCRIPT="scripts/organize_sweep_datasets.py"

# echo "=========================================="
# echo "Organizing Skywork DPO Datasets"
# echo "=========================================="
# python "$SCRIPT" \
#     --sweep_id "3e7zl14s" \
#     --loop_base_dir "$SCRATCH/ActiveUltraFeedback/datasets/loop/3e7zl14s" \
#     --output_base_dir "$SCRATCH/ActiveUltraFeedback/datasets/skywork/actives" \
#     --model_type "dpo"

echo ""
echo "=========================================="
echo "Organizing Skywork RM Datasets"
echo "=========================================="
# Note: You'll need to provide the correct sweep_id for Skywork RM
# The user mentioned qafq0hbz in their notes
python "$SCRIPT" \
    --sweep_id "qafq0hbz" \
    --loop_base_dir "$SCRATCH/ActiveUltraFeedback/datasets/loop/qafq0hbz" \
    --output_base_dir "$SCRATCH/ActiveUltraFeedback/datasets/skywork/actives" \
    --model_type "rm"

# echo ""
# echo "=========================================="
# echo "Organizing Combined DPO Datasets"
# echo "=========================================="
# python "$SCRIPT" \
#     --sweep_id "vvjrae7l" \
#     --loop_base_dir "$SCRATCH/ActiveUltraFeedback/datasets/loop/vvjrae7l" \
#     --output_base_dir "$SCRATCH/ActiveUltraFeedback/datasets/combined/actives" \
#     --model_type "dpo"

echo ""
echo "=========================================="
echo "Organizing Combined RM Datasets"
echo "=========================================="
python "$SCRIPT" \
    --sweep_id "kmmpoya5" \
    --loop_base_dir "$SCRATCH/ActiveUltraFeedback/datasets/loop/kmmpoya5" \
    --output_base_dir "$SCRATCH/ActiveUltraFeedback/datasets/combined/actives" \
    --model_type "rm"
