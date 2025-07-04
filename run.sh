#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --container-writable
#SBATCH --output=run_%j.out

srun --environment=activeuf python -u run.py --part 0
