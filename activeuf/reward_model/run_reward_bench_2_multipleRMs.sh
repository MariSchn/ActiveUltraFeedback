#!/bin/bash
# filepath: /iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/activeuf/reward_model/run_reward_bench_all.sh

# Array of reward model paths
MODELS=(
    # "$SCRATCH/models/reward_models/preference_albation_dts_llama_15"
    # "$SCRATCH/models/reward_models/preference_albation_dts_llama_16"
    # "$SCRATCH/models/reward_models/preference_albation_dts_llama_17"
    "$SCRATCH/models/reward_models/preference_albation_dts_llama_18"
    "$SCRATCH/models/reward_models/preference_albation_dts_llama_19"
    # "$SCRATCH/models/reward_models/preference_albation_dts_llama_20"
)

for MODEL in "${MODELS[@]}"; do
    echo "Running reward bench for model: $MODEL"
    bash $SCRATCH/ActiveUltraFeedback/activeuf/reward_model/run_reward_bench_2.sh "$MODEL"
done