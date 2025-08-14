#!/bin/bash
# filepath: /iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/activeuf/reward_model/run_reward_bench_all.sh

# Array of reward model paths
MODELS=(
    "$SCRATCH/models/reward_models/preference_new_new_dts_llama_2"
    "$SCRATCH/models/reward_models/preference_new_new_dts_llama_3"
    "$SCRATCH/models/reward_models/preference_new_dts_llama_1"
    "$SCRATCH/models/reward_models/preference_new_dts_llama_2"
    "$SCRATCH/models/reward_models/preference_new_dts_llama_3"
    "$SCRATCH/models/reward_models/preference_new_dts_llama_4"
    "$SCRATCH/models/reward_models/preference_new_dts_llama_5"
    "$SCRATCH/models/reward_models/preference_new_dts_llama_6"
    "$SCRATCH/models/reward_models/preference_new_dts_llama_7"
    "$SCRATCH/models/reward_models/preference_new_dts_llama_8"
    # Add more model paths as needed
)

for MODEL in "${MODELS[@]}"; do
    echo "Running reward bench for model: $MODEL"
    bash $SCRATCH/ActiveUltraFeedback/activeuf/reward_model/run_reward_bench_2.sh "$MODEL"
done