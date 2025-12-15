#!/bin/bash

#SBATCH -A a-infra01-1
#SBATCH --job-name=cpo
#SBATCH --output=logs/cpo_new_new/O-%x.%j
#SBATCH --error=logs/cpo_new_new/E-%x.%j
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=03:00:00
#SBATCH --environment=activeuf_dev

export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_DEBUG=INFO 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600

if [[ -n "$MyDatasetPath" ]]; then
    export DATASET_PATH="$MyDatasetPath"
    echo "Using Custom Dataset Path: $DATASET_PATH"
else
    export DATASET_PATH="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/my/actives/datasets/MaxMinLCB_60829"
    echo "Using Default Dataset Path: $DATASET_PATH"
fi

export HF_HOME=$SCRATCH/hf_home
export HF_TOKEN=$HF_TOKEN
export WANDB_PROJECT=CPO

export GPUS_PER_NODE=4
export MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MAIN_PROCESS_PORT=29500
export NUM_PROCESSES=$(expr $SLURM_NNODES \* $GPUS_PER_NODE)

START=$(date +%s)

echo "Starting training on $NUM_PROCESSES processes across $SLURM_NNODES nodes..."
echo "Master Node: $MAIN_PROCESS_IP"

# --- Dynamic Argument Construction ---
EXTRA_ARGS=""
if [[ -n "$MyLR" ]]; then EXTRA_ARGS="$EXTRA_ARGS --learning_rate $MyLR"; fi
if [[ -n "$MyBeta" ]]; then EXTRA_ARGS="$EXTRA_ARGS --beta $MyBeta"; fi
if [[ -n "$MyGamma" ]]; then EXTRA_ARGS="$EXTRA_ARGS --simpo_gamma $MyGamma"; fi
if [[ -n "$MyBS" ]]; then EXTRA_ARGS="$EXTRA_ARGS --per_device_train_batch_size $MyBS"; fi
if [[ -n "$MyGAS" ]]; then EXTRA_ARGS="$EXTRA_ARGS --gradient_accumulation_steps $MyGAS"; fi

if [[ "$MyUseLora" == "true" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --lora_r $MyRank --lora_alpha $MyAlpha"
fi

echo "Injecting Extra Args: $EXTRA_ARGS"

export CONFIG_FILE="$SCRATCH/ActiveUltraFeedback/configs/accelerate/deepspeed2.yaml"

srun bash -c '
pip install --upgrade trl "transformers==4.52.4" "tokenizers==0.21.4" && \
accelerate launch \
    --config_file="$CONFIG_FILE" \
    --num_processes "$NUM_PROCESSES" \
    --num_machines "$SLURM_NNODES" \
    --machine_rank "$SLURM_PROCID" \
    --main_process_ip "$MAIN_PROCESS_IP" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --rdzv_backend static \
    -m activeuf.cpo.training \
    --config_path "$SCRATCH/ActiveUltraFeedback/configs/cpo_training.yaml" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$SCRATCH/models/cpo" '"$EXTRA_ARGS"
    
END=$(date +%s)
DURATION=$(( END - START ))

echo "Job ended at: $(date)"
echo "Total execution time: $DURATION seconds"