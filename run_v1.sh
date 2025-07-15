#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --container-writable
#SBATCH --output=run_%j.out
#SBATCH --exclude=nid006438,nid006439,nid006440,nid006441,nid006442,nid006443,nid006444,nid006445,nid006446,nid006447,nid006448,nid006449,nid006450,nid006451,nid006461,nid006462,nid006868

srun --environment=activeuf python -u run_v1.py