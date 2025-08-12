#!/bin/bash

# Define variables for running the script:
CONFIG_FILE="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_gpu.yaml"
TRAINER_SCRIPT="activeuf.active_learning_loop"
COMPLETIONS_DATASET_PATH="$SCRATCH/datasets/combined_annotations_llama/"
OUTPUT_PATH="$SCRATCH/datasets/testssss/"
REPORT_TO="wandb"
ACQUISITION_FUNCTION_TYPE="dts"

export CUDA_VISIBLE_DEVICES=0,1,2,3

#running the script
accelerate launch --config_file="$CONFIG_FILE" -m "$TRAINER_SCRIPT" \
    --completions_dataset_path="$COMPLETIONS_DATASET_PATH" \
    --output_path="$OUTPUT_PATH" \
    --report_to="$REPORT_TO" \
    --acquisition_function_type="$ACQUISITION_FUNCTION_TYPE"