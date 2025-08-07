#!/bin/bash
#SBATCH -A a-infra01-1
#SBATCH --job-name=rm_training
#SBATCH --time=04:00:00
#SBATCH --environment=activeuf_dev
#SBATCH --ntasks-per-node=1         
#SBATCH --gres=gpu:4                
#SBATCH --cpus-per-task=288         
#SBATCH --output=./RM-Training/O-%x.%j
#SBATCH --error=./RM-Training/E-%x.%j

# Make sure HF_TOKEN variable is set to the Hugging Face token
export HF_TOKEN=$HF_TOKEN
export HF_HOME=$SCRATCH/huggingface

cd $SCRATCH/ActiveUltraFeedback/

# Define variables for paths
# dts qwen დამავიწყდა, და თურმე m_l-ში გავუშვი
CONFIG_FILE="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_gpu.yaml"
TRAINER_SCRIPT="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/reward_trainer.py"
OUTPUT_DIR="$SCRATCH/models/reward_models/preference_random_llama"
REWARD_CONFIG="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/reward_config.yaml"
DATASET_PATH="$SCRATCH/datasets/preference_random_llama/"

# Run the training command
accelerate launch --config_file="$CONFIG_FILE" "$TRAINER_SCRIPT" --output_dir="$OUTPUT_DIR" --reward_config="$REWARD_CONFIG" --dataset_path="$DATASET_PATH"