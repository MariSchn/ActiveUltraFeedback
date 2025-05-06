#!/bin/bash

# Define variables for paths
CONFIG_FILE="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_gpu.yaml"
TRAINER_SCRIPT="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/reward_trainer.py"
OUTPUT_DIR="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/trainedModels/firstTrainedModel"
REWARD_CONFIG="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/reward_config.yaml"

# Run the training command
accelerate launch --config_file="$CONFIG_FILE" "$TRAINER_SCRIPT" --output_dir="$OUTPUT_DIR" --reward_config="$REWARD_CONFIG"