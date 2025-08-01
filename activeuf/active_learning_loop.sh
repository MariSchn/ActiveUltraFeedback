#!/bin/bash

#SBATCH --job-name=ultrafeedback_llama_loop
#SBATCH -D .
#SBATCH -A a-infra01-1
#SBATCH --output=/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/loop/llama/ultrafeedback/O-%x.%j
#SBATCH --error=/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/loop/llama/ultrafeedback/E-%x.%j
#SBATCH --time=01:30:00             # maximum execution time (HH:MM:SS)
#SBATCH --environment=activeuf_dev      # using compressed docker image as an environment


start_time=$(date +%s)

acquisition_function="ultrafeedback"
annotator_model="llama"

accelerate launch \
    --config_file=$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_gpu.yaml \
    -m activeuf.active_learning_loop \
    --completions_dataset_path ${SCRATCH}/datasets/combined_annotations_${annotator_model}/ \
    --output_path=$SCRATCH/datasets/preference_${acquisition_function}_${annotator_model}/ \
    --logs_path=$SCRATCH/ActiveUltraFeedback/activeuf/logs/${annotator_model}_${acquisition_function}.log \
    --args_path=$SCRATCH/ActiveUltraFeedback/activeuf/logs/${annotator_model}_${acquisition_function}_args.log \
    --acquisition_config=$SCRATCH/ActiveUltraFeedback/activeuf/acquisition_function/configs.yaml \
    --acquisition_function_type=${acquisition_function}

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total execution time: $duration seconds"