#!/bin/bash

# Define variables for running the script:
CONFIG_FILE="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_gpu.yaml"
TRAINER_SCRIPT="activeuf.active_learning_loop"
COMPLETIONS_DATASET_PATH="$SCRATCH/datasets/combined_annotations_llama/"
OUTPUT_PATH="$SCRATCH/datasets/tested/"
REPORT_TO="none"
ACQUISITION_FUNCTION_TYPE="dts"
REGULARIZATION_TOWARDS_INITIAL_WEIGHTS=10.0
REGULARIZATION_WEIGHT_DECAY_TYPE="linear"
EXPONENTIAL_DECAY_BASE=0.95
MAX_TRAINING_STEPS=40
INITIALIZATION_XAVIER_GAIN=2.0
BASE_MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-1B-Instruct"

export CUDA_VISIBLE_DEVICES=0,1,2,3

#running the script
accelerate launch --config_file="$CONFIG_FILE" -m "$TRAINER_SCRIPT" \
    --completions_dataset_path="$COMPLETIONS_DATASET_PATH" \
    --output_path="$OUTPUT_PATH" \
    --report_to="$REPORT_TO" \
    --acquisition_function_type="$ACQUISITION_FUNCTION_TYPE" \


# --regularization_weight_decay_type="$REGULARIZATION_WEIGHT_DECAY_TYPE" \
# --initialization_xavier_gain="$INITIALIZATION_XAVIER_GAIN" \
# --base_model_name_or_path="$BASE_MODEL_NAME_OR_PATH" \
# --max_training_steps="$MAX_TRAINING_STEPS" \
# --base_model_name_or_path="$BASE_MODEL_NAME_OR_PATH" \
# --exponential_decay_base="$EXPONENTIAL_DECAY_BASE" \
# --regularization_weight_decay_type="$REGULARIZATION_WEIGHT_DECAY_TYPE" \
# --regularization_towards_initial_weights="$REGULARIZATION_TOWARDS_INITIAL_WEIGHTS"
