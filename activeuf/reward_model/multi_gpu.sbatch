#!/bin/bash
#SBATCH -A a-a145
#SBATCH --job-name=activeuf_rm_training
#SBATCH --time=01:30:00
#SBATCH --output=run_loop_%j.out
#SBATCH --environment=activeuf
#SBATCH --ntasks-per-node=1         
#SBATCH --gres=gpu:4                
#SBATCH --cpus-per-task=288         
#SBATCH --output=active_loop/O-%x.%j
#SBATCH --error=active_loop/E-%x.%j
#SBATCH --partition=debug

# Activate container environment
export HF_TOKEN=<Huggingface Token>
export HF_HOME=$SCRATCH/huggingface

cd $SCRATCH/ActiveUltraFeedback/

# Define variables for paths
CONFIG_FILE="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_gpu.yaml"
TRAINER_SCRIPT="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/reward_trainer.py"
OUTPUT_DIR="$SCRATCH/models/active_32_3200_after_5000_fair"
REWARD_CONFIG="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/reward_config.yaml"

# Run the training command
accelerate launch --config_file="$CONFIG_FILE" "$TRAINER_SCRIPT" --output_dir="$OUTPUT_DIR" --reward_config="$REWARD_CONFIG"