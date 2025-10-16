#!/bin/bash

#SBATCH --job-name=new_loop
#SBATCH -D .
#SBATCH -A a-infra01-1
#SBATCH --output=./logs/loop/O-%x.%j
#SBATCH --error=./logs/loop/E-%x.%j
#SBATCH --time=01:00:00
#SBATCH --environment=activeuf_dev
#SBATCH --partition=debug

python -m pip install -e resources/rewarduq

accelerate launch \
    --config_file=activeuf/new_loop/accelerate.yaml \
    -m activeuf.new_loop.run --config_path activeuf/new_loop/run.yaml