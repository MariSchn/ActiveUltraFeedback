#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <models_list_file>"
    exit 1
fi

MODELS_LIST_FILE="$1"
RESULTS_DIR="/iopsstor/scratch/cscs/dmelikidze/models/reward_models/results/"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

echo "results directory: $RESULTS_DIR"

MODELS=()
while IFS= read -r line; do
    MODELS+=("$line")
done < "$MODELS_LIST_FILE"

total=${#MODELS[@]}
count=0

for MODEL in "${MODELS[@]}"; do
    count=$((count + 1))
    # Get parent dir and checkpoint name for output file
    parent_dir=$(basename "$(dirname "$MODEL")")
    checkpoint_name=$(basename "$MODEL")
    result_file="${RESULTS_DIR}/${parent_dir}/${checkpoint_name}.json"

    # Create parent_dir inside RESULTS_DIR if it doesn't exist
    mkdir -p "${RESULTS_DIR}/${parent_dir}"

    if [ -f "$result_file" ]; then
        echo "[$count/$total] Skipping $MODEL: result file already exists."
        continue
    fi

    echo "[$count/$total] Running reward bench for model: $MODEL"
    bash $SCRATCH/ActiveUltraFeedback/activeuf/reward_model/run_reward_bench_2.sh "$MODEL"
done