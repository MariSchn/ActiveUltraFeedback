#!/bin/bash

#SBATCH --job-name=random_loop
#SBATCH -D .
#SBATCH -A a-infra01-1
#SBATCH --output=loop/llama/O-%x.%j
#SBATCH --error=loop/llama/E-%x.%j
#SBATCH --time=03:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --environment=activeuf_dev      # using compressed docker image as an environment

accelerate launch \
    --config_file=$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_gpu.yaml \
    -m activeuf.active_learning_loop \
    --completions_dataset_path ${SCRATCH}/datasets/combined_annotations_llama/ \
    --output_path=$SCRATCH/datasets/acquisition_function_random/ \
    --logs_path=$SCRATCH/ActiveUltraFeedback/activeuf/logs/llama_random \
    --args_path=$SCRATCH/ActiveUltraFeedback/activeuf/logs/llama_random_args \
    --acquisition_config=$SCRATCH/ActiveUltraFeedback/activeuf/acquisition_function/configs.yaml \
    --acquisition_function_type=random