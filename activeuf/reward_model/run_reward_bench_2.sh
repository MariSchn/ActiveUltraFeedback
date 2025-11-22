#!/bin/bash

# Usage:
# TODO Add submodule to repo (skip step 1 once submodule is added)
# 1. git clone https://github.com/allenai/reward-bench.git resources/reward-bench
# 2. chmod +x activeuf/reward_model/run_reward_bench_2.sh
# 3. pip install resources/reward-bench
# 4. Run it with your model name: activeuf/reward_model/run_reward_bench_2.sh /iopsstor/scratch/cscs/dmelikidze/models/reward_models/preference_albation_dts_llama_6__/checkpoint-7250-processed

# Check if model name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

MODEL_NAME=$1

# Run the Python script with the provided model
accelerate launch --config_file=$SCRATCH/ActiveUltraFeedback/configs/accelerate/single_node.yaml resources/reward-bench/scripts/run_v2.py --model="$MODEL_NAME" --max_length=4096