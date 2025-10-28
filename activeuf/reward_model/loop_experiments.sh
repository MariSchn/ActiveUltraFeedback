#!/bin/bash
set -euo pipefail

# This script submits reward-model training jobs (uses the same sbatch layout as multi_node.sbatch)
# It scans $SCRATCH/datasets/active and submits one job per selected dataset directory.
# By default it SUBSAMPLES the list using START_IDX..END_IDX (keeps previous behavior).
#
# To change which datasets are run, set environment variables before calling:
#   START_IDX (inclusive), END_IDX (exclusive)
# Example:
#   START_IDX=0 END_IDX=10 ./loop_experiments.sh

BASE_DATASETS_DIR="${SCRATCH:-/scratch}/datasets/active"
REWARD_MULTI_NODE_CFG="$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_node.yaml"
PYTHON_FILE="${SCRATCH}/ActiveUltraFeedback/activeuf/reward_model/reward_trainer.py"
REWARD_CONFIG="${SCRATCH}/ActiveUltraFeedback/activeuf/reward_model/reward_config.yaml"

# Subsampling defaults (previously used 22..23)
START_IDX=${START_IDX:-1}
END_IDX=${END_IDX:-24}

if [ ! -d "$BASE_DATASETS_DIR" ]; then
  echo "ERROR: datasets dir not found: $BASE_DATASETS_DIR" >&2
  exit 1
fi

FINAL_DATASETS=()
for DATASET_PATH in "$BASE_DATASETS_DIR"/*; do
  [ -d "$DATASET_PATH" ] || continue
  FINAL_DATASETS+=("$DATASET_PATH")
done

echo "Found ${#FINAL_DATASETS[@]} datasets in ${BASE_DATASETS_DIR}."

# clamp indices
if [ "$START_IDX" -lt 0 ]; then START_IDX=0; fi
if [ "$END_IDX" -le "$START_IDX" ]; then END_IDX=$(( START_IDX + 1 )); fi
if [ "$START_IDX" -ge "${#FINAL_DATASETS[@]}" ]; then
  echo "START_IDX ($START_IDX) >= number of datasets (${#FINAL_DATASETS[@]}). Nothing to do."
  exit 0
fi
if [ "$END_IDX" -gt "${#FINAL_DATASETS[@]}" ]; then
  END_IDX=${#FINAL_DATASETS[@]}
fi

SUBSAMPLE_DATASETS=()
for ((i=START_IDX; i<END_IDX && i<${#FINAL_DATASETS[@]}; i++)); do
    SUBSAMPLE_DATASETS+=("${FINAL_DATASETS[$i]}")
done
# echo "Subsampled dataset elements: ${SUBSAMPLE_DATASETS[@]}"
# echo "Subsampled ${#SUBSAMPLE_DATASETS[@]} datasets for training:"
# exit 0

echo "Subsampled ${#SUBSAMPLE_DATASETS[@]} datasets (indices ${START_IDX}..$((END_IDX-1)))."
for DATASET_PATH in "${SUBSAMPLE_DATASETS[@]}"; do
  [ -d "$DATASET_PATH" ] || continue

  RUN_NAME="$(basename "$DATASET_PATH")"
  JOB_NAME="rm_${RUN_NAME}"
  JOB_NAME="${JOB_NAME:0:80}"

  echo "Preparing job for dataset: $DATASET_PATH -> run_name: $RUN_NAME"

  sbatch <<EOF
#!/bin/bash
#SBATCH -A a-infra01-1
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=./RM-Training/O-${JOB_NAME}.%j
#SBATCH --error=./RM-Training/E-${JOB_NAME}.%j
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=03:30:00
#SBATCH --environment=activeuf_dev

export GPUS_PER_NODE=4
export HF_HOME=\$SCRATCH/huggingface
export HF_TOKEN=\$HF_TOKEN

# compute head node inside the job and reference it with an escaped var so expansion
# happens on the compute node (not when generating the heredoc)
head_node_ip=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)

export LAUNCHER="accelerate launch \
    --config_file=${REWARD_MULTI_NODE_CFG} \
    --num_processes \$((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines \$SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip \$head_node_ip \
    --main_process_port 29500 \
    "

export ACCELERATE_DIR="\${ACCELERATE_DIR:-/accelerate}"
export PYTHON_FILE="${PYTHON_FILE}"

# build output dir under SCRATCH/models/reward_models using the dataset folder name
# export OUTPUT_DIR="\${SCRATCH}/models/reward_models/\${RUN_NAME}"
# mkdir -p "\${OUTPUT_DIR}"

export SCRIPT_ARGS=" \
    --output_dir \${SCRATCH}/models/reward_models/${RUN_NAME} \
    --reward_config ${REWARD_CONFIG} \
    --dataset_path ${DATASET_PATH} \
    "

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="\$LAUNCHER \$PYTHON_FILE \$SCRIPT_ARGS"

START=\$(date +%s)
cd \$SCRATCH/ActiveUltraFeedback/
srun \$CMD
END=\$(date +%s)
DURATION=\$(( END - START ))

echo "Job ended at: \$(date)"
echo "Total execution time: \$DURATION seconds"
EOF

  sleep 0.5
done

echo "All jobs submitted."