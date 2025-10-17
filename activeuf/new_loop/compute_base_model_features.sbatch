#!/bin/bash

#SBATCH --job-name=compute_base_model_features
#SBATCH -D .
#SBATCH -A a-infra01-1
#SBATCH --output=./logs/loop/O-%x.%j
#SBATCH --error=./logs/loop/E-%x.%j
#SBATCH --time=8:00:00
#SBATCH --environment=activeuf_dev
#SBATCH --partition=normal

NCCL_DEBUG=ERROR

python -m pip install -e resources/rewarduq

accelerate launch \
    --config_file=activeuf/new_loop/accelerate.yaml \
    -m activeuf.new_loop.compute_base_model_features --config_path activeuf/new_loop/compute_base_model_features.yaml