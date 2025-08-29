#!/bin/bash
# filepath: /iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/activeuf/reward_model/run_reward_bench_all.sh

# Array of reward model paths
MODELS=(
    # "$SCRATCH/models/reward_models/preference_ablation_dts_llama_1_new"
    # "$SCRATCH/models/reward_models/preference_ablation_our_judge_54_our"
    # "$SCRATCH/models/reward_models/preference_ablation_random_1_new"
    # "$SCRATCH/models/reward_models/preference_ablation_ultrafeedback_1_new"
    # "$SCRATCH/models/reward_models/preference_albation_dts_llama_6_new"
    # "$SCRATCH/models/reward_models/preference_ablation_our_judge_54"
    # "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/skywork_allenai"
    "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/dts_1"
    "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/dts_2"
    "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/dts_3"
    "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/dts_4"
    "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/dts_5"
    "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/skywork_10000"
    "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/skywork_20000"
    "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/skywork_30000"
    "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/skywork_50000"
    "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/skywork_60000"
    "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/skywork_70000"
    "/iopsstor/scratch/cscs/dmelikidze/models/reward_models/skywork_78000"
)

for MODEL in "${MODELS[@]}"; do
    echo "Running reward bench for model: $MODEL"
    bash $SCRATCH/ActiveUltraFeedback/activeuf/reward_model/run_reward_bench_2.sh "$MODEL"
done