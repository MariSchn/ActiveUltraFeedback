#!/bin/bash

#SBATCH -A a-infra01-1
#SBATCH --job-name=dpo
#SBATCH --output=logs/dpo/O-%x.%j
#SBATCH --error=logs/dpo/E-%x.%j
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=09:00:00             # maximum execution time (HH:MM:SS)

BASE_CACHE_DIR="$SCRATCH"
DATASET_PATH="<DATASET_PATH>"
SEED="<SEED>"

export HF_HOME=$BASE_CACHE_DIR/hf_home
export VLLM_CACHE_DIR=$BASE_CACHE_DIR/vllm_cache
export WANDB_PROJECT=DPO
export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"

# below is based on https://docs.csc.fi/support/tutorials/ml-multi/accelerate
MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MAIN_PROCESS_PORT=29500
NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_ON_NODE)

CMD="accelerate launch \
    --config_file=$SCRATCH/ActiveUltraFeedback/configs/accelerate/deepspeed2.yaml \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_NNODES \
    --machine_rank \$SLURM_NODEID \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $MAIN_PROCESS_PORT \
    -m activeuf.dpo.training \
    --config_path $SCRATCH/ActiveUltraFeedback/configs/dpo_training.yaml \
    --slurm_job_id $SLURM_JOB_ID \
    --dataset_path $DATASET_PATH \
    --beta 0.1 \
    --learning_rate 2e-5 \
    --seed $SEED \
    --num_epochs 3"

echo $CMD

START=$(date +%s)

srun --environment=activeuf_dev bash -c "$CMD"

END=$(date +%s)
DURATION=$(( END - START ))

echo "Job ended at: $(date)"
echo "Total execution time: $DURATION seconds"