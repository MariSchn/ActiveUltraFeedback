#!/bin/bash
#SBATCH --job-name=full_run
#SBATCH -A a-infra01-1
#SBATCH --output=./logs/loop/full_run_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --nodes=4

# TODO: STORE ENV VARS SCRIPT IN LOOP OUTPUT DIR
# TODO: REFERENCE DPO AND RM RUNS IN LOOP RUN (CONFIG IN WANDB)


echo -e "======================="
echo -e "===== ECHO SCRIPT ====="
echo -e "=======================\n\n\n"

cat ./scripts/loop_train_eval.sbatch

echo -e "\n\n\n==========================="
echo -e "===== STARTING SCRIPT ====="
echo -e "==========================="




echo -e "\n\n\n========================"
echo -e "===== SHARED SETUP ====="
echo -e "========================\n\n\n"


set -eo pipefail

export WANDB_ENTITY=ActiveUF

export OLMES_SCRIPT_PATH="${SCRATCH}/ActiveUltraFeedback/resources/olmes/installation/unattended-eval.sh"

export BASE_RM_OUTPUT_DIR="${SCRATCH}/models/reward_models"
export BASE_DPO_OUTPUT_DIR="${SCRATCH}/models/dpo"
export WANDB_DIR="${SCRATCH}/cache/wandb"
export HF_HOME="${SCRATCH}/cache/hf_cache"

export LOOP_CONFIG="./configs/loop.yaml"
export DPO_CONFIG="./configs/dpo_training.yaml"
export RM_CONFIG="./configs/rm_training.yaml"
export RM_TRAINING_ACCELERATE_CONFIG="./configs/accelerate/deepspeed2.yaml"
export DPO_TRAINING_ACCELERATE_CONFIG="./configs/accelerate/deepspeed2.yaml"
export ACCELERATE_CONFIG="./configs/accelerate/multi_node.yaml"

export GPUS_PER_NODE=4
export NUM_PROCESSES=$(expr "$SLURM_NNODES" \* "$GPUS_PER_NODE")
export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"
export NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
export HEAD_NODE=${NODELIST[0]}
export HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -i)
export HEAD_PROCESS_PORT=29500
export SWEEP_ID=$WANDB_SWEEP_ID

echo "Environment variables set:"
echo "WANDB_ENTITY=${WANDB_ENTITY}"
echo "BASE_RM_OUTPUT_DIR=${BASE_RM_OUTPUT_DIR}"
echo "BASE_DPO_OUTPUT_DIR=${BASE_DPO_OUTPUT_DIR}"
echo "WANDB_DIR=${WANDB_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "LOOP_CONFIG=${LOOP_CONFIG}"
echo "DPO_CONFIG=${DPO_CONFIG}"
echo "RM_CONFIG=${RM_CONFIG}"
echo "ACCELERATE_CONFIG=${ACCELERATE_CONFIG}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "ACCELERATE_DIR=${ACCELERATE_DIR}"
echo "NODELIST=${NODELIST[@]}"
echo "HEAD_NODE=${HEAD_NODE}"
echo "HEAD_NODE_IP=${HEAD_NODE_IP}"
echo "HEAD_PROCESS_PORT=${HEAD_PROCESS_PORT}"
echo "SWEEP_ID=${SWEEP_ID}"






echo -e "\n\n\n================================"
echo -e "===== ACTIVE LEARNING LOOP ====="
echo -e "================================\n\n\n"

export WANDB_PROJECT=loop
export TEMP_VARS_FILE="./.tmp/loop_vars_${SLURM_JOB_ID}.sh"  # ! This needs to be synced with the run.py script

# * "$@" passes all CLI args to the script, which is used for sweeps
export LOOP_LAUNCHER="accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_JOB_NUM_NODES \
    --rdzv_backend c10d \
    --main_process_ip $HEAD_NODE_IP \
    --main_process_port $HEAD_PROCESS_PORT \
    "
export LOOP_ARGS="--config_path $LOOP_CONFIG $@"
export LOOP_CMD="$LOOP_LAUNCHER -m activeuf.loop.run $LOOP_ARGS"

echo "Running command: $LOOP_CMD"
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --environment=activeuf_dev --output=./logs/loop/loop_%j.out $LOOP_CMD

# Env vars "LOOP_WANDB_RUN_ID", "LOOP_DATASET_PATH", "SWEEP_ID" are set in the $TEMP_VARS_FILE which is written by the loop run script
if [ ! -f "$TEMP_VARS_FILE" ]; then
    echo "Error: $TEMP_VARS_FILE not found!"
    exit 1
fi
source "$TEMP_VARS_FILE"
rm "$TEMP_VARS_FILE"

echo "Completed Active Learning Loop."
echo "LOOP_WANDB_RUN_ID=${LOOP_WANDB_RUN_ID}"
echo "LOOP_DATASET_PATH=${LOOP_DATASET_PATH}"
echo "SWEEP_ID=${SWEEP_ID}"

wait

echo -e "\n\n\n========================"
echo -e "===== DPO TRAINING ====="
echo -e "========================\n\n\n"

# Unset sweep-related environment variables to prevent DPO training from reusing the loop's WandB run
unset WANDB_SWEEP_ID
unset WANDB_RUN_ID

if [ -n "$SWEEP_ID" ]; then
    export DPO_OUTPUT_DIR="${BASE_DPO_OUTPUT_DIR}/${SWEEP_ID}/${LOOP_WANDB_RUN_ID}"
else
    export WANDB_PROJECT=DPO
    export DPO_OUTPUT_DIR="${BASE_DPO_OUTPUT_DIR}/${LOOP_WANDB_RUN_ID}"
fi
export DPO_NNODES=2
export DPO_NUM_PROCESSES=$(expr $DPO_NNODES \* $GPUS_PER_NODE)
export DPO_NODELIST="${NODELIST[0]},${NODELIST[1]}"
export DPO_HEAD_NODE=${NODELIST[0]}
export DPO_HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$DPO_HEAD_NODE" hostname -i)
export DPO_HEAD_PROCESS_PORT=29500

echo "Starting DPO Training on nodes ${NODELIST[0]},${NODELIST[1]}"
echo "DPO_OUTPUT_DIR=${DPO_OUTPUT_DIR}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"

DPO_TRAIN_LAUNCHER="accelerate launch \
    --config_file $DPO_TRAINING_ACCELERATE_CONFIG \
    --num_processes $DPO_NUM_PROCESSES \
    --num_machines $DPO_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $DPO_HEAD_NODE_IP \
    --main_process_port $DPO_HEAD_PROCESS_PORT \
    "
DPO_TRAIN_ARGS=" \
    --config_path $DPO_CONFIG \
    --slurm_job_id $SLURM_JOB_ID \
    --dataset_path $LOOP_DATASET_PATH \
    --output_dir $DPO_OUTPUT_DIR \
    "
DPO_TRAIN_CMD="$DPO_TRAIN_LAUNCHER -m activeuf.dpo.training $DPO_TRAIN_ARGS"

echo "Running command: $DPO_TRAIN_CMD"
srun --nodes=$DPO_NNODES --ntasks=$DPO_NNODES --ntasks-per-node=1 --nodelist=$DPO_NODELIST --output=${DPO_OUTPUT_DIR}/training.log --environment=activeuf_dev $DPO_TRAIN_CMD &


echo -e "\n\n\n================================="
echo -e "===== REWARD MODEL TRAINING ====="
echo -e "=================================\n\n\n"

# Unset sweep-related environment variables to prevent RM training from reusing the loop's WandB run
unset WANDB_SWEEP_ID
unset WANDB_RUN_ID

if [ -n "$SWEEP_ID" ]; then
    export RM_OUTPUT_DIR="${BASE_RM_OUTPUT_DIR}/${SWEEP_ID}/${LOOP_WANDB_RUN_ID}"
else
    export WANDB_PROJECT=RM
    export RM_OUTPUT_DIR="${BASE_RM_OUTPUT_DIR}/${LOOP_WANDB_RUN_ID}"
fi
export RM_NNODES=2
export RM_NUM_PROCESSES=$(expr $RM_NNODES \* $GPUS_PER_NODE)
export RM_NODELIST="${NODELIST[2]},${NODELIST[3]}"
export RM_HEAD_NODE=${NODELIST[2]}
export RM_HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$RM_HEAD_NODE" hostname -i)
export RM_HEAD_PROCESS_PORT=29501 # Different from DPO_HEAD_PROCESS_PORT to be safe

echo "Starting RM Training on nodes ${NODELIST[2]},${NODELIST[3]}"
echo "RM_OUTPUT_DIR=${RM_OUTPUT_DIR}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"

RM_TRAIN_LAUNCHER="accelerate launch \
    --config_file $RM_TRAINING_ACCELERATE_CONFIG \
    --num_processes $RM_NUM_PROCESSES \
    --num_machines $RM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $RM_HEAD_NODE_IP \
    --main_process_port $RM_HEAD_PROCESS_PORT"
RM_TRAIN_ARGS=" \
    --output_dir $RM_OUTPUT_DIR \
    --reward_config $RM_CONFIG \
    --dataset_path $LOOP_DATASET_PATH"
RM_TRAIN_CMD="$RM_TRAIN_LAUNCHER ./activeuf/reward_model/training.py $RM_TRAIN_ARGS"

echo "Running command: $RM_TRAIN_CMD"
srun --nodes=$RM_NNODES --ntasks=$RM_NNODES --ntasks-per-node=1 --nodelist=$RM_NODELIST --output=${RM_OUTPUT_DIR}/training.log --environment=activeuf_dev $RM_TRAIN_CMD &

echo "Waiting for DPO and RM training to complete..."
wait
echo "DPO and RM training complete."




echo -e "\n\n\n=========================="
echo -e "===== DPO EVALUATION ====="
echo -e "===========================\n\n\n"

echo "Starting DPO Evaluation"
echo "Evaluating DPO model at: $DPO_OUTPUT_DIR"
mkdir -p "${DPO_OUTPUT_DIR}/results/gsm8k_tulu" \
         "${DPO_OUTPUT_DIR}/results/ifeval_tulu" \
         "${DPO_OUTPUT_DIR}/results/truthfulqa_tulu"


export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PROJECT_ROOT_AT=$SCRATCH/projects/ActiveUltraFeedback/resources/olmes
export PROJECT_NAME=olmes
export PACKAGE_NAME=oe_eval
export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
export WANDB_API_KEY_FILE_AT=$HOME/.wandb-api-key
export HF_HOME=$SCRATCH/cache/hf_cache
export SKIP_INSTALL_PROJECT=1
export SHARED=/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CONTAINER_IMAGES=/capstor/store/cscs/swissai/infra01/container-images
unset SSL_CERT_FILE

CONTAINER_ARGS=" \
  --container-image=$CONTAINER_IMAGES/infra01+ismayilz+olmes+arm64-cuda-root-latest.sqsh \
  --environment="${PROJECT_ROOT_AT}/installation/edf.toml" \
  --container-mounts=\
$PROJECT_ROOT_AT,\
$SCRATCH,\
/iopsstor/scratch/cscs/smoalla/projects/dpr/,\
$SHARED,\
$WANDB_API_KEY_FILE_AT,\
$HOME/.gitconfig,\
$HOME/.bashrc,\
$HOME/.ssh \
  --container-workdir=$PROJECT_ROOT_AT \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
"

export DPO_EVAL_ARGS=" \
    --model=${LOOP_WANDB_RUN_ID} \
    --model-wb-name=${LOOP_WANDB_RUN_ID} \
    --model-type=vllm \
    --batch-size=1 \
    --model-args '{\"tensor_parallel_size\": 4, \"max_length\": 4096, \"add_bos_token\": false, \"model_path\": \"${DPO_OUTPUT_DIR}\", \"trust_remote_code\": true}' \
    --use-chat-format=True \
    "

echo "Starting gsm8k (on ${NODELIST[0]})"
srun --output=${DPO_OUTPUT_DIR}/results/gsm8k_tulu/log.out \
     --nodes=1 --ntasks=1 --gpus-per-task=4 -w "${NODELIST[0]}" \
     ${CONTAINER_ARGS} \
     bash -c "exec python3 -m oe_eval.launch --task=gsm8k::tulu --output-dir=${DPO_OUTPUT_DIR}/results/gsm8k_tulu ${DPO_EVAL_ARGS}" &

# echo "Starting minerva_math (on ${NODELIST[1]})"
# srun --output=${DPO_OUTPUT_DIR}/results/minerva_math_tulu/log.out \
#      --nodes=1 --ntasks=1 --gpus-per-task=4 -w "${NODELIST[1]}" \
#      ${CONTAINER_ARGS} \
#      bash -c "exec python3 -m oe_eval.launch --task=minerva_math::tulu --output-dir=${DPO_OUTPUT_DIR}/results/minerva_math_tulu ${DPO_EVAL_ARGS}" &

echo "Starting ifeval (on ${NODELIST[2]})"
srun --output=${DPO_OUTPUT_DIR}/results/ifeval_tulu/log.out \
     --nodes=1 --ntasks=1 --gpus-per-task=4 -w "${NODELIST[2]}" \
     ${CONTAINER_ARGS} \
     bash -c "exec python3 -m oe_eval.launch --task=ifeval::tulu --output-dir=${DPO_OUTPUT_DIR}/results/ifeval_tulu ${DPO_EVAL_ARGS}" &

echo "Starting truthfulqa (on ${NODELIST[3]})"
srun --output=${DPO_OUTPUT_DIR}/results/truthfulqa_tulu/log.out \
     --nodes=1 --ntasks=1 --gpus-per-task=4 -w "${NODELIST[3]}" \
     ${CONTAINER_ARGS} \
     bash -c "exec python3 -m oe_eval.launch --task=truthfulqa::tulu --output-dir=${DPO_OUTPUT_DIR}/results/truthfulqa_tulu ${DPO_EVAL_ARGS}" &

echo "All DPO evals launched."





echo -e "\n\n\n===================================="
echo -e "===== REWARD MODEL EVALUATION ====="
echo -e "===================================\n\n\n"

echo "Starting RM Evaluation"
echo "Evaluating RM at: $RM_OUTPUT_DIR"

echo "Running command: bash ./activeuf/reward_model/reward_bench_2.sh --model $RM_OUTPUT_DIR"
srun --output=${RM_OUTPUT_DIR}/eval_%j.out \
     --environment=activeuf_dev \
     --nodes=1 --ntasks=1 --nodelist="${NODELIST[1]}" \
     bash ./activeuf/reward_model/reward_bench_2.sh --model $RM_OUTPUT_DIR &

# Wait for ALL background jobs (4 DPO evals + 1 RM eval) to finish
echo "Waiting for all DPO and RM evaluations to complete..."
wait
echo "All evaluations finished."

echo -e "\n\n\n===================================="
echo -e "===== UPDATE METRICS IN WANDB ======"
echo -e "====================================\n\n\n"

srun --environment=activeuf_dev --nodes=1 --ntasks=1 python ./scripts/update_wandb_run.py --run_id ${LOOP_WANDB_RUN_ID} --rm_output_dir ${RM_OUTPUT_DIR} --dpo_output_dir ${DPO_OUTPUT_DIR} --project loop --entity ${WANDB_ENTITY}

echo "Script completed successfully."