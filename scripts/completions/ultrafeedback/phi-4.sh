#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --time=6:00:00
#SBATCH --container-writable
#SBATCH --job-name=phi-4
#SBATCH --output=./logs/phi-4_%j.out
#SBATCH --exclude=nid006438,nid006439,nid006440,nid006441,nid006442,nid006443,nid006444,nid006445,nid006446,nid006447,nid006448,nid006449,nid006450,nid006451,nid006461,nid006462,nid006868

srun --environment=activeuf_06_25 python -m activeuf.generate_completions --dataset_path /iopsstor/scratch/cscs/smarian/datasets/allenai/ultrafeedback_binarized_cleaned/train_prefs --model_name microsoft/phi-4 --output_path /iopsstor/scratch/cscs/smarian/datasets/completions/phi-4
