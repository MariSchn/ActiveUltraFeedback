#!/bin/bash

#SBATCH --job-name=multinode_loop
#SBATCH -D .
#SBATCH -A infra01
#SBATCH --output=logs/experiments_new_new/O.%j
#SBATCH --error=logs/experiments_new_new/E.%j
#SBATCH --nodes=8                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=288         # number of cores per tasks
#SBATCH --time=2:15:00             # maximum execution time (HH:MM:SS)
#SBATCH --environment=activeuf_dev      # using compressed docker image as an environment
#SBATCH --partition=normal

export GPUS_PER_NODE=4
export HF_HOME=$SCRATCH/huggingface
export HF_TOKEN=$HF_TOKEN
######################

echo $ACCELERATE_DIR

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --config_file=$SCRATCH/ActiveUltraFeedback/configs/accelerate/multi_node.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    -m
    "

export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"
export PYTHON_FILE="activeuf.active_learning_loop"
if [ -z "${SCRIPT_ARGS:-}" ]; then
    export SCRIPT_ARGS=" \
    --completions_dataset_path /iopsstor/scratch/cscs/dmelikidze/datasets/combined_with_small_qwen_3_235b-features \
    --output_path=$SCRATCH/datasets/testssss/ \
        --acquisition_function_type=dts \
        --log_kpis \
        --report_to=wandb \
        --use_features \
    "
fi
        
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $SCRIPT_ARGS" 

export DS_ACCELERATOR_CACHE_DIR="$SCRATCH/tmp/deepspeed_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$SCRATCH/tmp/hf_cache_$SLURM_JOB_ID"
mkdir -p "$DS_ACCELERATOR_CACHE_DIR" "$TRANSFORMERS_CACHE"

START=$(date +%s)

cd $SCRATCH/ActiveUltraFeedback/
export PYTHONPATH="$SCRATCH/ActiveUltraFeedback:$PYTHONPATH"
pip install --user ./resources/rewarduq/

export CUDA_VISIBLE_DEVICES=0,1,2,3

srun --chdir=$SCRATCH/ActiveUltraFeedback $CMD

END=$(date +%s)
DURATION=$(( END - START ))

echo "Job ended at: $(date)"
echo "Total execution time: $DURATION seconds"