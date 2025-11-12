#!/bin/bash
#SBATCH --job-name=dist-hw-test
#SBATCH --account=a-infra01-1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --output=./logs/test_hardware_driver.out
#SBATCH --environment=activeuf_dev
#SBATCH --exclusive

GPUS_PER_NODE=4
SCRIPT_PATH=scripts/docker/test_hardware_driver.py
MASTER_PORT=${MASTER_PORT:-29505}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}

set -euo pipefail

echo "SLURM_JOBID        = $SLURM_JOB_ID"
echo "SLURM_NODELIST     = $SLURM_JOB_NODELIST"
echo "SLURM_NNODES       = $SLURM_NNODES"
echo "SLURM_PROCS (total)= $SLURM_NTASKS"

# Derive MASTER_ADDR as the first node hostname
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

echo "Torch / NCCL debug vars:"
env | grep -E 'MASTER_|NCCL_|TORCH_DIST' || true

# Sanity: ensure script exists
if [[ ! -f $SCRIPT_PATH ]]; then
  echo "Error: SCRIPT_PATH $SCRIPT_PATH not found" >&2
  exit 1
fi

echo "Starting distributed test across $SLURM_NNODES nodes x $GPUS_PER_NODE GPUs"

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
  set -euo pipefail
  echo "[Node $SLURM_NODEID] launching torchrun (rank launcher) on host $(hostname)"
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node='"$GPUS_PER_NODE"' \
    --node-rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    '"$SCRIPT_PATH"'
'

status=$?
if [[ $status -eq 0 ]]; then
  echo "Distributed hardware communication sanity check succeeded." 
else
  echo "Distributed hardware communication sanity check FAILED with status $status" >&2
fi

exit $status
