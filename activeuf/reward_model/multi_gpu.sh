#!/bin/bash

# Define variables for paths
CONFIG_FILE="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_gpu.yaml"
TRAINER_SCRIPT="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/reward_trainer.py"
OUTPUT_DIR="$SCRATCH/models/reward_models/preference_random_llama/"
REWARD_CONFIG="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/reward_config.yaml"
DATASET_PATH="$SCRATCH/datasets/preference_random_llama/"

# Run the training command
accelerate launch --config_file="$CONFIG_FILE" "$TRAINER_SCRIPT" --output_dir="$OUTPUT_DIR" --reward_config="$REWARD_CONFIG" --dataset_path="$DATASET_PATH"