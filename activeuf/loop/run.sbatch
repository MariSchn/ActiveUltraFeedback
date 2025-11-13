#!/bin/bash

#SBATCH --job-name=new_loop
#SBATCH -D .
#SBATCH -A a-infra01-1
#SBATCH --output=./logs/loop/O-%x.%j
#SBATCH --error=./logs/loop/E-%x.%j
#SBATCH --time=5:00:00
#SBATCH --environment=activeuf_dev
#SBATCH --partition=normal

accelerate launch \
    --config_file=configs/accelerate/single_node.yaml \
    -m activeuf.loop.run --config_path configs/loop.yaml