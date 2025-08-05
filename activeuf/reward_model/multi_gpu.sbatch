#!/bin/bash
#SBATCH -A a-infra01-1
#SBATCH --job-name=activeuf_rm_training
#SBATCH --time=08:00:00
#SBATCH --output=run_loop_%j.out
#SBATCH --environment=activeuf_dev
#SBATCH --ntasks-per-node=1         
#SBATCH --gres=gpu:4                
#SBATCH --cpus-per-task=288         
#SBATCH --output=active_loop_m_l/O-%x.%j
#SBATCH --error=active_loop_m_l/E-%x.%j

# Activate container environment
export HF_TOKEN=<HF_TOKEN>
export HF_HOME=$SCRATCH/huggingface

cd $SCRATCH/ActiveUltraFeedback/

# Define variables for paths
# dts qwen დამავიწყდა, და თურმე m_l-ში გავუშვი
CONFIG_FILE="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_gpu.yaml"
TRAINER_SCRIPT="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/reward_trainer.py"
OUTPUT_DIR="$SCRATCH/models/reward_models/preference_ultrafeedback_llama"
REWARD_CONFIG="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/reward_config.yaml"
DATASET_PATH="$SCRATCH/datasets/preference_ultrafeedback_llama/"

# Run the training command
accelerate launch --config_file="$CONFIG_FILE" "$TRAINER_SCRIPT" --output_dir="$OUTPUT_DIR" --reward_config="$REWARD_CONFIG" --dataset_path="$DATASET_PATH"