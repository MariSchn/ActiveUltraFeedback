#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --container-writable
#SBATCH --job-name=OLMo-2-0325-32B-Instruct
#SBATCH --output=./logs/OLMo-2-0325-32B-Instruct_%j.out
#SBATCH --exclude=nid006438,nid006439,nid006440,nid006441,nid006442,nid006443,nid006444,nid006445,nid006446,nid006447,nid006448,nid006449,nid006450,nid006451,nid006461,nid006462,nid006868

srun --environment=activeuf_06_25 python -m activeuf.generate_completions --dataset_path /iopsstor/scratch/cscs/smarian/datasets/allenai/ultrafeedback_binarized_cleaned/train_prefs --model_name allenai/OLMo-2-0325-32B-Instruct --output_path /iopsstor/scratch/cscs/smarian/datasets/completions/OLMo-2-0325-32B-Instruct
