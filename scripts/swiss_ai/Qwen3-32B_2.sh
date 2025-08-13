#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --container-writable
#SBATCH --job-name=Qwen3-32B_2 
#SBATCH --output=./logs/swiss_ai/Qwen3-32B_2_%j.out
#SBATCH --exclude=nid006438,nid006439,nid006440,nid006441,nid006442,nid006443,nid006444,nid006445,nid006446,nid006447,nid006448,nid006449,nid006450,nid006451,nid006461,nid006462,nid006868

srun --environment=activeuf_dev python swiss_ai.py --dataset_path /iopsstor/scratch/cscs/smarian/datasets/swiss_ai/olmo_split2 --model_name Qwen/Qwen3-32B --output_path /iopsstor/scratch/cscs/smarian/datasets/swiss_ai/raw/Qwen3-32B_2