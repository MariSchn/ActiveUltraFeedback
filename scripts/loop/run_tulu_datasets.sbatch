#!/bin/bash
#SBATCH --job-name=tuple_run
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=./logs/looptulu/tuple_run_%A_%a.out  # %A=JobId, %a=ArrayIdx
#SBATCH --error=./logs/looptulu/tuple_run_%A_%a.err
#SBATCH --cpus-per-task=288

# ==========================================
# ===== 1. CONFIGURATION ===================
# ==========================================

# Defaults (Hardcoded as requested)
DATASET_PATH="datasets/tulu_3/qwen_3_235b"
OUTER_BS=64
ENN_BS=64
SAVE_STEPS=250

# Select the Combo ID automatically from the Slurm Array Index
# If running locally (not sbatch), default to 0
COMBO_ID=${SLURM_ARRAY_TASK_ID:-0}

echo -e "\n==========================="
echo -e "===== JOB ARRAY STATUS ===="
echo -e "==========================="
echo "Slurm Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Current Task ID:    ${SLURM_ARRAY_TASK_ID}"
echo "Running Combo ID:   ${COMBO_ID}"
echo "Dataset Path:       ${DATASET_PATH}"
echo "Save Every:         ${SAVE_STEPS} steps"

# ==========================================
# ===== 2. HYPERPARAMETER TUPLES ===========
# ==========================================

case $COMBO_ID in
  # --- GROUP 1 (Previously Commented) ---
  0)
    ACQ_TYPE="drts" 
    TAG="checkpoint"
    CKPT="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_3/actives/dpo/drts_enn_qwen_ultrafeedback_20260106-144557-020/checkpoint-checkpoint-3500"
    
    TUPLE_ARGS_ARRAY=(
        "--acquisition_function_type=drts"
        "--acquisition_function.beta=1.0"
        "--enn.regularization.exponential_decay_base=0.999"
        "--enn.regularization.initial_value=1.0"
        "--replay_buffer_factor=1000"
    )
    ;;
  1)    
    ACQ_TYPE="dts"
    TAG="checkpoint"
    CKPT="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_3/actives/dpo/dts_enn_qwen_ultrafeedback_20260106-144556-725/checkpoint-checkpoint-3500"
    
    TUPLE_ARGS_ARRAY=(
        "--acquisition_function_type=dts"
        "--acquisition_function.beta=1.0"
        "--enn.regularization.exponential_decay_base=0.99"
        "--enn.regularization.initial_value=1.0"
        "--replay_buffer_factor=1000"
        "--resume_from_checkpoint=$CKPT"
    )
    ;;
  2)
    ACQ_TYPE="deltaucb"
    TAG="checkpoint"
    CKPT="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_3/actives/dpo/deltaucb_enn_qwen_ultrafeedback_20260106-144555-800/checkpoint-checkpoint-3500"
    
    TUPLE_ARGS_ARRAY=(
        "--acquisition_function_type=deltaucb"
        "--acquisition_function.beta=2.0"
        "--enn.regularization.exponential_decay_base=0.999"
        "--enn.regularization.initial_value=1.0"
        "--replay_buffer_factor=1000"
        "--resume_from_checkpoint=$CKPT"
    )
    ;;
  3)
    ACQ_TYPE="infomax"
    TAG="checkpoint"
    CKPT="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_3/actives/dpo/infomax_enn_qwen_ultrafeedback_20260106-144559-775/checkpoint-checkpoint-3500"
    
    TUPLE_ARGS_ARRAY=(
        "--acquisition_function_type=infomax"
        "--acquisition_function.beta=1.0"
        "--enn.regularization.exponential_decay_base=0.99"
        "--enn.regularization.initial_value=1.0"
        "--replay_buffer_factor=1000"
        "--resume_from_checkpoint=$CKPT"
    )
    ;;
  4)
    ACQ_TYPE="maxminlcb"
    TAG="checkpoint"
    CKPT="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_3/actives/dpo/maxminlcb_enn_qwen_ultrafeedback_20260106-150110-615/checkpoint-checkpoint-3500"
    
    TUPLE_ARGS_ARRAY=(
        "--acquisition_function_type=maxminlcb"
        "--acquisition_function.beta=1.0"
        "--enn.regularization.exponential_decay_base=0.99"
        "--enn.regularization.initial_value=1.0"
        "--replay_buffer_factor=1000"
        "--resume_from_checkpoint=$CKPT"
    )
    ;;

  # --- GROUP 2 (Previously Active) ---
  # 0)
  #   ACQ_TYPE="drts"
  #   TAG="drts"
  #   CKPT="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_chkp_rm/drts"
    
  #   TUPLE_ARGS_ARRAY=(
  #       "--acquisition_function_type=drts"
  #       "--acquisition_function.beta=1.0"
  #       "--enn.regularization.exponential_decay_base=0.9"
  #       "--enn.regularization.initial_value=1.0"
  #       "--replay_buffer_factor=1000"
  #   )
  #   ;;
  # 1)
  #   ACQ_TYPE="dts"
  #   TAG="dts_b1.0_d0.999_r1000"
  #   CKPT="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_3/actives/dpo/dts_enn_qwen_ultrafeedback_20260106-151550-694/checkpoint-dts_b1.0_d0.999_r1000-3500"
    
  #   TUPLE_ARGS_ARRAY=(
  #       "--acquisition_function_type=dts"
  #       "--acquisition_function.beta=1.0"
  #       "--enn.regularization.exponential_decay_base=0.999"
  #       "--enn.regularization.initial_value=1.0"
  #       "--replay_buffer_factor=1000"
  #       "--resume_from_checkpoint=$CKPT"
  #   )
  #   ;;
  # 2)
  #   ACQ_TYPE="deltaucb"
  #   TAG="deltaucb_b1.0_d0.99_r100"
  #   CKPT="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_3/actives/dpo/deltaucb_enn_qwen_ultrafeedback_20260106-151650-907/checkpoint-deltaucb_b1.0_d0.99_r100-4250"
    
  #   TUPLE_ARGS_ARRAY=(
  #       "--acquisition_function_type=deltaucb"
  #       "--acquisition_function.beta=1.0"
  #       "--enn.regularization.exponential_decay_base=0.99"
  #       "--enn.regularization.initial_value=1.0"
  #       "--replay_buffer_factor=100"
  #       "--resume_from_checkpoint=$CKPT"
  #   )
  #   ;;
  # 3)
  #   ACQ_TYPE="infomax"
  #   TAG="infomax"
  #   CKPT="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_chkp_rm/infomax"
    
  #   TUPLE_ARGS_ARRAY=(
  #       "--acquisition_function_type=infomax"
  #       "--acquisition_function.beta=2.0"
  #       "--enn.regularization.exponential_decay_base=0.99"
  #       "--enn.regularization.initial_value=1.0"
  #       "--replay_buffer_factor=1000"
  #       "--resume_from_checkpoint=$CKPT"
  #   )
  #   ;;
  # 4)
  #   ACQ_TYPE="maxminlcb"
  #   TAG="maxminlcb"
  #   CKPT="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_chkp_rm/maxminlcb"
    
  #   TUPLE_ARGS_ARRAY=(
  #       "--acquisition_function_type=maxminlcb"
  #       "--acquisition_function.beta=2.0"
  #       "--enn.regularization.exponential_decay_base=0.9"
  #       "--enn.regularization.initial_value=1.0"
  #       "--replay_buffer_factor=1000"
  #   )
  #   ;;
  *)
    echo "Error: Invalid ID $COMBO_ID"
    exit 1
    ;;
esac

# === CRITICAL STEP: FLATTEN THE ARRAY ===
# This converts the multi-line array into a single long string with spaces.
# This ensures 'bash -c' receives one continuous command with no hidden newlines.
TUPLE_ARGS="${TUPLE_ARGS_ARRAY[*]}"

echo "Applying Parameters ($TAG): $TUPLE_ARGS"
# 1. Force NCCL to use the correct network interface
export NCCL_SOCKET_IFNAME=hsn

# 2. Fix for AWS-OFI-NCCL on Cray Slingshot 11
export FI_PROVIDER=cxi
export NCCL_NET_GDR_LEVEL=PHB  # Changed from 3 - PHB is more reliable on Slingshot

# 3. Prevent CPU Contention (CRITICAL for Accelerate inside Srun)
# Without this, each python process might try to grab ALL CPU cores, 
# causing the node to freeze during initialization.
export OMP_NUM_THREADS=4

# 4. NCCL Timeout and Debug Settings (CRITICAL for diagnosing hangs)
export NCCL_DEBUG=WARN                    # Set to INFO for more verbose debugging
export NCCL_DEBUG_SUBSYS=INIT,NET         # Focus on init and network issues
export NCCL_TIMEOUT=1800                  # 30 min timeout instead of infinite hang
export NCCL_IB_TIMEOUT=23                 # Infiniband timeout
export NCCL_ASYNC_ERROR_HANDLING=1        # Enable async error handling

# 5. Torch Distributed Timeout
export TORCH_DISTRIBUTED_DEBUG=DETAIL     # Helps diagnose where hangs occur
export TORCH_NCCL_BLOCKING_WAIT=0         # Non-blocking wait to prevent deadlocks

# 6. Use dynamic port to avoid conflicts from stale processes
export HEAD_PROCESS_PORT=$((29500 + RANDOM % 1000))

# ==========================================
# ===== 3. ENVIRONMENT SETUP ===============
# ==========================================

set -eo pipefail
export WANDB_ENTITY=ActiveUF
echo "wandb entity: $WANDB_ENTITY"
export LOOP_CONFIG="./configs/loop.yaml"
echo "loop config: $LOOP_CONFIG"
export ACCELERATE_CONFIG="./configs/accelerate/multi_node.yaml"
echo "accelerate config: $ACCELERATE_CONFIG"
export PYTHONPATH="$(pwd):$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"
export WANDB_DIR="${SCRATCH}/cache/wandb"
echo "wandb dir: $WANDB_DIR"
export HF_HOME="${SCRATCH}/cache/hf_cache"
echo "HF cache dir: $HF_HOME"

MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "Main Process IP: $MAIN_PROCESS_IP"
# HEAD_PROCESS_PORT is now set dynamically above to avoid conflicts
echo "Main Process Port: $HEAD_PROCESS_PORT"
export NUM_PROCESSES=$(expr "$SLURM_JOB_NUM_NODES" \* "4") 
echo "Number of Processes: $NUM_PROCESSES"

echo "HEAD_IP: $MAIN_PROCESS_IP"
echo "TOTAL_GPUS: $NUM_PROCESSES"
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT"
echo "NCCL_DEBUG: $NCCL_DEBUG"

# ==========================================
# ===== 4. EXECUTION =======================
# ==========================================

export WANDB_PROJECT=loop

CONSTANTS_ARRAY=(
    "--inputs_path=$DATASET_PATH"
    "--outer_loop_batch_size=$OUTER_BS"
    "--enn.effective_batch_size=$ENN_BS"
    "--save_every_n_outer_batches=$SAVE_STEPS"
    "--run_tag=$TAG"
    "--run_id=$ACQ_TYPE"
)
CONSTANTS="${CONSTANTS_ARRAY[*]}"

LOOP_ARGS="--config_path $LOOP_CONFIG $TUPLE_ARGS $CONSTANTS --resume_from_checkpoint=$CKPT"

# The command string is now safe because TUPLE_ARGS contains no newlines
# Using c10d backend instead of static for more robust rendezvous
# Adding small staggered delay inside the command to let master start first
CMD="sleep \$((SLURM_PROCID * 2)) && accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_JOB_NUM_NODES \
    --machine_rank \$SLURM_PROCID \
    --rdzv_backend c10d \
    --rdzv_endpoint $MAIN_PROCESS_IP:$HEAD_PROCESS_PORT \
    --rdzv_conf timeout=1800 \
    -m activeuf.loop.run $LOOP_ARGS"

echo "Running distributed command..."
echo $CMD

srun --nodes=$SLURM_JOB_NUM_NODES \
     --ntasks=$SLURM_JOB_NUM_NODES \
     --ntasks-per-node=1 \
     --environment=activeuf \
     bash -c "$CMD"