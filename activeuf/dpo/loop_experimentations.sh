#!/bin/bash
set -euo pipefail

# Scan datasets and submit a DPO job per dataset. Each dataset folder name is used
# verbatim as the DPO run name:
#   {acquisition}_{annotator}_rgl{regularizer}_wdcb{decay_base}_obs{outer_loop_batch}_rbs{replay_buffer_size}_{slurm_job_id}
#
# The script only builds run names and submits jobs with --run_name set to the folder name.

BASE_DATASETS_DIR="${SCRATCH:-/scratch}/datasets/active"
DPO_CONFIG_PATH="$SCRATCH/ActiveUltraFeedback/activeuf/dpo/training.yaml"
MULTI_NODE_CFG="$SCRATCH/ActiveUltraFeedback/activeuf/dpo/multi_node.yaml"
ACCELERATE_LAUNCH_BASE="accelerate launch --config_file=${MULTI_NODE_CFG} -m activeuf.dpo.training"

if [ ! -d "$BASE_DATASETS_DIR" ]; then
  echo "ERROR: datasets dir not found: $BASE_DATASETS_DIR" >&2
  exit 1
fi

FINAL_DATASETS=()

for DATASET_PATH in "$BASE_DATASETS_DIR"/*; do
  [ -d "$DATASET_PATH" ] || continue
  FINAL_DATASETS+=("$DATASET_PATH")
done

# echo "Dataset elements: ${FINAL_DATASETS[@]}"
echo "Found ${#FINAL_DATASETS[@]} datasets to process."

SUBSAMPLE_DATASETS=()
for ((i=22; i<23 && i<${#FINAL_DATASETS[@]}; i++)); do
    SUBSAMPLE_DATASETS+=("${FINAL_DATASETS[$i]}")
done

echo "Subsampled dataset elements: ${SUBSAMPLE_DATASETS[@]}"
echo "Subsampled ${#SUBSAMPLE_DATASETS[@]} datasets for training:"
exit 0
for DATASET_PATH in "${SUBSAMPLE_DATASETS[@]}"; do
  [ -d "$DATASET_PATH" ] || continue

  RUN_NAME="$(basename "$DATASET_PATH")"
  # ensure RUN_NAME is safe for job names (max length and remove problematic chars)
  JOB_NAME="dpo_${RUN_NAME}"
  # truncate long job names to 80 chars to be safe
  JOB_NAME="${JOB_NAME:0:80}"

  echo "Preparing job for dataset: $DATASET_PATH -> run_name: $RUN_NAME"

  sbatch <<EOF
#!/bin/bash
#SBATCH -A a-infra01-1
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/dpo/O-${JOB_NAME}.%j
#SBATCH --error=logs/dpo/E-${JOB_NAME}.%j
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=03:00:00

export HF_HOME=\$SCRATCH/hf_home
export VLLM_CACHE_DIR=\$SCRATCH/vllm_cache
export WANDB_PROJECT=DPO
export ACCELERATE_DIR="\${ACCELERATE_DIR:-/accelerate}"

MAIN_PROCESS_IP=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
MAIN_PROCESS_PORT=29500
NUM_PROCESSES=\$(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))

CMD="accelerate launch \\
    --config_file=${MULTI_NODE_CFG} \\
    --num_processes \$NUM_PROCESSES \\
    --num_machines \$SLURM_NNODES \\
    --machine_rank \\\$SLURM_NODEID \\
    --main_process_ip \$MAIN_PROCESS_IP \\
    --main_process_port \$MAIN_PROCESS_PORT \\
    -m activeuf.dpo.training \\
    --config_path ${DPO_CONFIG_PATH} \\
    --slurm_job_id \$SLURM_JOB_ID \\
    --dataset_path ${DATASET_PATH} \\
    --beta 0.1 \\
    --learning_rate 2e-5 \\
    --seed 4 \\
    --num_epochs 3"
    
echo \$CMD
START=\$(date +%s)
srun --chdir=\$SCRATCH/ActiveUltraFeedback --environment=activeuf_dev bash -lc "\$CMD"
END=\$(date +%s)
DURATION=\$(( END - START ))
echo "Job ended at: \$(date) (duration=\$DURATION s)"
EOF

  # small throttle to avoid overloading the scheduler
  sleep 0.5
done

echo "All jobs submitted."