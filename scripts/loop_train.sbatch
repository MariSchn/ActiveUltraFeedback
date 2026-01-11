#!/bin/bash
#SBATCH --job-name=tuple_run
#SBATCH -A a-infra01-1
#SBATCH --output=./logs/loop/tuple_run_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --nodes=8

# ==========================================
# ===== 1. ARGUMENT PARSING ================
# ==========================================

# Defaults
COMBO_ID=0
DATASET_PATH="datasets/ultrafeedback/qwen_3_235b"
OUTER_BS=64
ENN_BS=64

# Loop through all args passed by WandB
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --combo_id=*) COMBO_ID="${1#*=}" ;;
        --combo_id) COMBO_ID="$2"; shift ;;
        --inputs_path=*) DATASET_PATH="${1#*=}" ;;
        --inputs_path) DATASET_PATH="$2"; shift ;;
        
        --outer_loop_batch_size=*) OUTER_BS="${1#*=}" ;;
        --outer_loop_batch_size) OUTER_BS="$2"; shift ;;
        --enn.effective_batch_size=*) ENN_BS="${1#*=}" ;;
        --enn.effective_batch_size) ENN_BS="$2"; shift ;;

        *) ;; 
    esac
    shift
done

echo -e "\n\n\n==========================="
echo -e "===== TUPLE SELECTION ====="
echo -e "==========================="
echo "Selected Combination ID: $COMBO_ID"
echo "Selected Dataset Path:   $DATASET_PATH"

# ==========================================
# ===== 2. HYPERPARAMETER TUPLES ===========
# ==========================================

# FIX: Use key=value format for all arguments below
case $COMBO_ID in
  # 0)
  #   TUPLE_ARGS="--acquisition_function_type=drts \
  #               --acquisition_function.beta=1.0 \
  #               --enn.regularization.exponential_decay_base=0.999 \
  #               --enn.regularization.initial_value=1.0 \
  #               --replay_buffer_factor=250"
  #   ;;
  # 1)
  #   TUPLE_ARGS="--acquisition_function_type=dts \
  #               --acquisition_function.beta=1.0 \
  #               --enn.regularization.exponential_decay_base=0.99 \
  #               --enn.regularization.initial_value=1.0 \
  #               --replay_buffer_factor=250"
  #   ;;
  # 2)
  #   TUPLE_ARGS="--acquisition_function_type=deltaucb \
  #               --acquisition_function.beta=2.0 \
  #               --enn.regularization.exponential_decay_base=0.999 \
  #               --enn.regularization.initial_value=1.0 \
  #               --replay_buffer_factor=250"
  #   ;;
  # 3)
  #   TUPLE_ARGS="--acquisition_function_type=infomax \
  #               --acquisition_function.beta=1.0 \
  #               --enn.regularization.exponential_decay_base=0.99 \
  #               --enn.regularization.initial_value=1.0 \
  #               --replay_buffer_factor=250"
  #   ;;
  # 4)
  #   TUPLE_ARGS="--acquisition_function_type=maxminlcb \
  #               --acquisition_function.beta=1.0 \
  #               --enn.regularization.exponential_decay_base=0.99 \
  #               --enn.regularization.initial_value=1.0 \
  #               --replay_buffer_factor=250"
  #   ;;
  0)
    TUPLE_ARGS="--acquisition_function_type=drts \
                --acquisition_function.beta=1.0 \
                --enn.regularization.exponential_decay_base=0.9 \
                --enn.regularization.initial_value=1.0 \
                --replay_buffer_factor=1000"
    ;;
  1)
    TUPLE_ARGS="--acquisition_function_type=dts \
                --acquisition_function.beta=1.0 \
                --enn.regularization.exponential_decay_base=0.999 \
                --enn.regularization.initial_value=1.0 \
                --replay_buffer_factor=1000"
    ;;
  2)
    TUPLE_ARGS="--acquisition_function_type=deltaucb \
                --acquisition_function.beta=1.0 \
                --enn.regularization.exponential_decay_base=0.99 \
                --enn.regularization.initial_value=1.0 \
                --replay_buffer_factor=100"
    ;;
  3)
    TUPLE_ARGS="--acquisition_function_type=infomax \
                --acquisition_function.beta=2.0 \
                --enn.regularization.exponential_decay_base=0.99 \
                --enn.regularization.initial_value=1.0 \
                --replay_buffer_factor=1000"
    ;;
  4)
    TUPLE_ARGS="--acquisition_function_type=maxminlcb \
                --acquisition_function.beta=2.0 \
                --enn.regularization.exponential_decay_base=0.9 \
                --enn.regularization.initial_value=1.0 \
                --replay_buffer_factor=1000"
    ;;
  *)
    echo "Error: Invalid combo_id $COMBO_ID"
    exit 1
    ;;
esac

echo "Applying Parameters: $TUPLE_ARGS"

# ==========================================
# ===== 3. ENVIRONMENT SETUP ===============
# ==========================================

echo -e "\n\n\n========================"
echo -e "===== SHARED SETUP ====="
echo -e "========================\n\n\n"

set -eo pipefail

export WANDB_ENTITY=ActiveUF
export LOOP_CONFIG="./configs/loop.yaml"
export ACCELERATE_CONFIG="./configs/accelerate/multi_node.yaml"

# Add current directory to python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Other Exports
export OLMES_SCRIPT_PATH="${SCRATCH}/ActiveUltraFeedback/resources/olmes/installation/unattended-eval.sh"
export BASE_RM_OUTPUT_DIR="${SCRATCH}/models/reward_models"
export BASE_DPO_OUTPUT_DIR="${SCRATCH}/models/dpo"
export WANDB_DIR="${SCRATCH}/cache/wandb"
export HF_HOME="${SCRATCH}/cache/hf_cache"

export GPUS_PER_NODE=4
if [ -z "$SLURM_NNODES" ]; then
    export SLURM_NNODES=1
fi
export NUM_PROCESSES=$(expr "$SLURM_NNODES" \* "$GPUS_PER_NODE")
export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"
export NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
export HEAD_NODE=${NODELIST[0]}
export HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -i)
export HEAD_PROCESS_PORT=29500
export SWEEP_ID=$WANDB_SWEEP_ID

echo "HEAD_NODE_IP=${HEAD_NODE_IP}"
echo "SLURM_NNODES=${SLURM_NNODES}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "SWEEP_ID=${SWEEP_ID}"

# ==========================================
# ===== 4. EXECUTION =======================
# ==========================================

echo -e "\n\n\n================================"
echo -e "===== ACTIVE LEARNING LOOP ====="
echo -e "================================\n\n\n"

export WANDB_PROJECT=loop
export TEMP_VARS_FILE="./.tmp/loop_vars_${SLURM_JOB_ID}.sh"

export LOOP_LAUNCHER="accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $HEAD_NODE_IP \
    --main_process_port $HEAD_PROCESS_PORT \
    "

# FIX: Use key=value format here as well
CONSTANTS="--inputs_path=$DATASET_PATH \
           --outer_loop_batch_size=$OUTER_BS \
           --enn.effective_batch_size=$ENN_BS"

export LOOP_ARGS="--config_path $LOOP_CONFIG $TUPLE_ARGS $CONSTANTS"
export LOOP_CMD="$LOOP_LAUNCHER -m activeuf.loop.run $LOOP_ARGS"

echo "Running command: $LOOP_CMD"

# Removed --output redirection so you can see errors in WandB logs
srun --nodes=$SLURM_NNODES \
     --ntasks=$SLURM_NNODES \
     --ntasks-per-node=1 \
     --environment=activeuf_dev \
     $LOOP_CMD