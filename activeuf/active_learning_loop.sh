#!/bin/bash

#SBATCH --job-name=loop_llama
#SBATCH -D .
#SBATCH -A a-infra01-1
#SBATCH --output=loop/llama/O-%x.%j
#SBATCH --error=loop/llama/E-%x.%j
#SBATCH --time=01:30:00             # maximum execution time (HH:MM:SS)
#SBATCH --environment=activeuf_dev      # using compressed docker image as an environment

pip install jax

accelerate launch \
    --config_file=$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_gpu.yaml \
    -m activeuf.active_learning_loop \
    --completions_dataset_path ${SCRATCH}/datasets/combined_annotations_llama/ \
    --output_path=$SCRATCH/datasets/testssss/ \
    --logs_path=$SCRATCH/logs_final_test_llama \
    --args_path=$SCRATCH/models_enn_test_llama \
    --acquisition_config=$SCRATCH/ActiveUltraFeedback/activeuf/acquisition_function/configs.yaml \
    --report_to="wandb"