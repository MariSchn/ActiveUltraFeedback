#!/bin/bash

#SBATCH -A a-infra01-1
#SBATCH --job-name=cpo
#SBATCH --output=logs/cpo/O-%x.%j
#SBATCH --error=logs/cpo/E-%x.%j
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=05:00:00
#SBATCH --environment=activeuf_dev

BASE_CACHE_DIR="$SCRATCH"
DATASET_PATH="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/my/actives/datasets/MaxMinLCB_60829"

export HF_HOME=$BASE_CACHE_DIR/hf_home
# Remove VLLM if not strictly used, or point to scratch
export WANDB_PROJECT=CPO
export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"

# --- FIX 2: PREVENT FILESYSTEM DEADLOCKS (CRITICAL) ---
# Redirect JIT compilation caches to local /tmp instead of shared network drive
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_${SLURM_JOB_ID}
export HF_DATASETS_CACHE=$BASE_CACHE_DIR/hf_datasets_cache

# Create these directories locally on the compute nodes
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "mkdir -p $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR"

MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MAIN_PROCESS_PORT=29500
NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_ON_NODE)

START=$(date +%s)

echo "Starting training on $NUM_PROCESSES processes across $SLURM_NNODES nodes..."
echo "Master Node: $MAIN_PROCESS_IP"

srun bash -c "
pip install git+https://github.com/huggingface/transformers.git && \
pip install git+https://github.com/huggingface/trl.git && \
unset SSL_CERT_FILE && \
accelerate launch \
    --config_file=$SCRATCH/ActiveUltraFeedback/configs/accelerate/deepspeed2.yaml \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_NNODES \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $MAIN_PROCESS_PORT \
    --rdzv_backend static \
    -m activeuf.cpo.training \
    --config_path $SCRATCH/ActiveUltraFeedback/configs/cpo_training.yaml \
    --dataset_path $DATASET_PATH \
    --output_dir $SCRATCH/models/cpo
"

END=$(date +%s)
DURATION=$(( END - START ))

echo "Job ended at: $(date)"
echo "Total execution time: $DURATION seconds"