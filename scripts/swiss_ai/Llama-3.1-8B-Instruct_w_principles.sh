#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --container-writable
#SBATCH --job-name=Llama-3.1-8B-Instruct_w_principles
#SBATCH --output=./logs/swiss_ai/Llama-3.1-8B-Instruct_w_principles_%j.out
#SBATCH --exclude=nid006438,nid006439,nid006440,nid006441,nid006442,nid006443,nid006444,nid006445,nid006446,nid006447,nid006448,nid006449,nid006450,nid006451,nid006461,nid006462,nid006868

srun --environment=activeuf_dev python swiss_ai.py --w_principles --dataset_path allenai/olmo-2-0325-32b-preference-mix --model_name meta-llama/Llama-3.1-8B-Instruct --output_path /iopsstor/scratch/cscs/smarian/datasets/swiss_ai/raw/Llama-3.1-8B-Instruct_w_principles
