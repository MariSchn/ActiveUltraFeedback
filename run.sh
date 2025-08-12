#!/bin/bash
#SBATCH --account=a-a10
#SBATCH --partition=debug
#SBATCH --time=01:10:00
#SBATCH --container-writable
#SBATCH --job-name=annotation_generation_0
#SBATCH --output=run_%j.out

srun --environment=activeuf python -u run.py --part 0
