#!/bin/bash
#SBATCH --job-name=tulu_run
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --time=12:00:00
#SBATCH --output=./logs/looptulu/tulu_run_%j.out
#SBATCH --error=./logs/looptulu/tulu_run_%j.err
#SBATCH --cpus-per-task=288

# ==========================================
# ===== SIMPLIFIED SINGLE JOB SCRIPT =======
# ==========================================
# This script expects these environment variables to be set:
#   ACQ_TYPE  - Acquisition function type (drts, dts, deltaucb, infomax, maxminlcb)
#   TAG       - Run tag for wandb
#   CKPT      - Checkpoint path
#   BETA      - Beta parameter
#   DECAY     - Exponential decay base
#   REPLAY    - Replay buffer factor
#
# Submit with:
#   sbatch --export=ALL,ACQ_TYPE=drts,TAG=test,CKPT=/path/to/ckpt,BETA=1.0,DECAY=0.99,REPLAY=1000 run_tulu_single.sbatch
# ==========================================

set -eo pipefail

echo "[$(date +%H:%M:%S)] Starting ${ACQ_TYPE} (Job ${SLURM_JOB_ID})"

# Validate required variables
if [[ -z "$ACQ_TYPE" || -z "$TAG" || -z "$CKPT" || -z "$BETA" || -z "$DECAY" || -z "$REPLAY" ]]; then
    echo "ERROR: Missing required environment variables!"
    exit 1
fi

DATASET_PATH="datasets/tulu_3/qwen_3_235b"
OUTER_BS=64
ENN_BS=64
SAVE_STEPS=250

# ==========================================
# ===== NCCL & NETWORK SETUP ===============
# ==========================================
export NCCL_SOCKET_IFNAME=hsn
export FI_PROVIDER=cxi
export NCCL_NET_GDR_LEVEL=PHB
export OMP_NUM_THREADS=4

# Timeouts and debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=23
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_BLOCKING_WAIT=0

HEAD_PROCESS_PORT=$((29500 + SLURM_JOB_ID % 1000))

# ==========================================
# ===== ENVIRONMENT SETUP ==================
# ==========================================
export WANDB_ENTITY=ActiveUF
export WANDB_PROJECT=loop
export LOOP_CONFIG="./configs/loop.yaml"
export ACCELERATE_CONFIG="./configs/accelerate/multi_node.yaml"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export WANDB_DIR="${SCRATCH}/cache/wandb"
export HF_HOME="${SCRATCH}/cache/hf_cache"

MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
NUM_PROCESSES=$((SLURM_JOB_NUM_NODES * 4))

# ==========================================
# ===== BUILD COMMAND ======================
# ==========================================

# Build arguments as a simple string (no arrays)
LOOP_ARGS="--config_path $LOOP_CONFIG"
LOOP_ARGS="$LOOP_ARGS --acquisition_function_type=$ACQ_TYPE"
LOOP_ARGS="$LOOP_ARGS --acquisition_function.beta=$BETA"
LOOP_ARGS="$LOOP_ARGS --enn.regularization.exponential_decay_base=$DECAY"
LOOP_ARGS="$LOOP_ARGS --enn.regularization.initial_value=1.0"
LOOP_ARGS="$LOOP_ARGS --replay_buffer_factor=$REPLAY"
LOOP_ARGS="$LOOP_ARGS --inputs_path=$DATASET_PATH"
LOOP_ARGS="$LOOP_ARGS --outer_loop_batch_size=$OUTER_BS"
LOOP_ARGS="$LOOP_ARGS --enn.effective_batch_size=$ENN_BS"
LOOP_ARGS="$LOOP_ARGS --save_every_n_outer_batches=$SAVE_STEPS"
LOOP_ARGS="$LOOP_ARGS --run_tag=$TAG"
LOOP_ARGS="$LOOP_ARGS --run_id=$ACQ_TYPE"
LOOP_ARGS="$LOOP_ARGS --resume_from_checkpoint=$CKPT"

echo "[$(date +%H:%M:%S)] Checking connectivity..."
if ! timeout 90 srun --nodes=$SLURM_JOB_NUM_NODES \
     --ntasks=$SLURM_JOB_NUM_NODES \
     --ntasks-per-node=1 \
     --environment=activeuf \
     bash -c 'true' 2>/dev/null; then
    echo "ERROR: One or more nodes did not respond"
    exit 1
fi

echo "[$(date +%H:%M:%S)] All nodes OK. Starting..."

# Build command
CMD="accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_JOB_NUM_NODES \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $HEAD_PROCESS_PORT \
    -m activeuf.loop.run $LOOP_ARGS"

srun --nodes=$SLURM_JOB_NUM_NODES \
     --ntasks=$SLURM_JOB_NUM_NODES \
     --ntasks-per-node=1 \
     --environment=activeuf \
     bash -c "$CMD"
