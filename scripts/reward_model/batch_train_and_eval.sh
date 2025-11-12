#!/bin/bash

readonly PROJECT_BASE_DIR="/iopsstor/scratch/cscs/smarian/projects/ActiveUltraFeedback"
readonly HF_HOME="/iopsstor/scratch/cscs/smarian/hf_cache"

readonly ACCELERATE_CONFIG="${PROJECT_BASE_DIR}/configs/accelerate/multi_node.yaml"
readonly PYTHON_FILE="${PROJECT_BASE_DIR}/activeuf/reward_model/training.py"
readonly REWARD_CONFIG="${PROJECT_BASE_DIR}/configs/rm_training.yaml"

readonly DATASETS_BASE_DIR="/iopsstor/scratch/cscs/smarian/datasets/7_preference_datasets"
readonly MODELS_BASE_DIR="/iopsstor/scratch/cscs/smarian/models/reward_models"

readonly RUNS=(
    # SKYWORK
    "skywork_with_small/llama_3.3_70b/max_min"
    "skywork_with_small/llama_3.3_70b/fixed_seed/100/random"
    "skywork_with_small/llama_3.3_70b/fixed_seed/100/ultrafeedback"
    "skywork_with_small/llama_3.3_70b/fixed_seed/101/random"
    "skywork_with_small/llama_3.3_70b/fixed_seed/101/ultrafeedback"
    "skywork_with_small/llama_3.3_70b/fixed_seed/102/random"
    "skywork_with_small/llama_3.3_70b/fixed_seed/102/ultrafeedback"
    "skywork_with_small/llama_3.3_70b/fixed_seed/103/random"
    "skywork_with_small/llama_3.3_70b/fixed_seed/103/ultrafeedback"
    "skywork_with_small/llama_3.3_70b/fixed_seed/104/random"
    "skywork_with_small/llama_3.3_70b/fixed_seed/104/ultrafeedback"
    "skywork_with_small/qwen_3_235b/fixed_seed/100/random"
    "skywork_with_small/qwen_3_235b/fixed_seed/100/ultrafeedback"
    "skywork_with_small/qwen_3_235b/fixed_seed/101/random"
    "skywork_with_small/qwen_3_235b/fixed_seed/101/ultrafeedback"
    "skywork_with_small/qwen_3_235b/fixed_seed/102/random"
    "skywork_with_small/qwen_3_235b/fixed_seed/102/ultrafeedback"
    "skywork_with_small/qwen_3_235b/fixed_seed/103/random"
    "skywork_with_small/qwen_3_235b/fixed_seed/103/ultrafeedback"
    "skywork_with_small/qwen_3_235b/fixed_seed/104/random"
    "skywork_with_small/qwen_3_235b/fixed_seed/104/ultrafeedback"
    "skywork_with_small/llama_8b_skywork_rm/fixed_seed/100/random"
    "skywork_with_small/llama_8b_skywork_rm/fixed_seed/100/ultrafeedback"
    "skywork_with_small/llama_8b_skywork_rm/fixed_seed/101/random"
    "skywork_with_small/llama_8b_skywork_rm/fixed_seed/101/ultrafeedback"
    "skywork_with_small/llama_8b_skywork_rm/fixed_seed/102/random"
    "skywork_with_small/llama_8b_skywork_rm/fixed_seed/102/ultrafeedback"
    "skywork_with_small/llama_8b_skywork_rm/fixed_seed/103/random"
    "skywork_with_small/llama_8b_skywork_rm/fixed_seed/103/ultrafeedback"
    "skywork_with_small/llama_8b_skywork_rm/fixed_seed/104/random"
    "skywork_with_small/llama_8b_skywork_rm/fixed_seed/104/ultrafeedback"

    # ULTRAFEEDBACK
    "ultrafeedback_with_small/llama_3.3_70b/max_min"
    "ultrafeedback_with_small/llama_3.3_70b/fixed_seed/100/random"
    "ultrafeedback_with_small/llama_3.3_70b/fixed_seed/100/ultrafeedback"
    "ultrafeedback_with_small/llama_3.3_70b/fixed_seed/101/random"
    "ultrafeedback_with_small/llama_3.3_70b/fixed_seed/101/ultrafeedback"
    "ultrafeedback_with_small/llama_3.3_70b/fixed_seed/102/random"
    "ultrafeedback_with_small/llama_3.3_70b/fixed_seed/102/ultrafeedback"
    "ultrafeedback_with_small/llama_3.3_70b/fixed_seed/103/random"
    "ultrafeedback_with_small/llama_3.3_70b/fixed_seed/103/ultrafeedback"
    "ultrafeedback_with_small/llama_3.3_70b/fixed_seed/104/random"
    "ultrafeedback_with_small/llama_3.3_70b/fixed_seed/104/ultrafeedback"
    "ultrafeedback_with_small/qwen_3_235b/max_min"
    "ultrafeedback_with_small/qwen_3_235b/fixed_seed/100/random"
    "ultrafeedback_with_small/qwen_3_235b/fixed_seed/100/ultrafeedback"
    "ultrafeedback_with_small/qwen_3_235b/fixed_seed/101/random"
    "ultrafeedback_with_small/qwen_3_235b/fixed_seed/101/ultrafeedback"
    "ultrafeedback_with_small/qwen_3_235b/fixed_seed/102/random"
    "ultrafeedback_with_small/qwen_3_235b/fixed_seed/102/ultrafeedback"
    "ultrafeedback_with_small/qwen_3_235b/fixed_seed/103/random"
    "ultrafeedback_with_small/qwen_3_235b/fixed_seed/103/ultrafeedback"
    "ultrafeedback_with_small/qwen_3_235b/fixed_seed/104/random"
    "ultrafeedback_with_small/qwen_3_235b/fixed_seed/104/ultrafeedback"
    "ultrafeedback_with_small/llama_8b_skywork_rm/max_min"
    "ultrafeedback_with_small/llama_8b_skywork_rm/fixed_seed/100/random"
    "ultrafeedback_with_small/llama_8b_skywork_rm/fixed_seed/100/ultrafeedback"
    "ultrafeedback_with_small/llama_8b_skywork_rm/fixed_seed/101/random"
    "ultrafeedback_with_small/llama_8b_skywork_rm/fixed_seed/101/ultrafeedback"
    "ultrafeedback_with_small/llama_8b_skywork_rm/fixed_seed/102/random"
    "ultrafeedback_with_small/llama_8b_skywork_rm/fixed_seed/102/ultrafeedback"
    "ultrafeedback_with_small/llama_8b_skywork_rm/fixed_seed/103/random"
    "ultrafeedback_with_small/llama_8b_skywork_rm/fixed_seed/103/ultrafeedback"
    "ultrafeedback_with_small/llama_8b_skywork_rm/fixed_seed/104/random"
    "ultrafeedback_with_small/llama_8b_skywork_rm/fixed_seed/104/ultrafeedback"

    # COMBINED
    "combined_with_small/llama_3.3_70b/max_min"
    "combined_with_small/llama_3.3_70b/fixed_seed/100/random"
    "combined_with_small/llama_3.3_70b/fixed_seed/100/ultrafeedback"
    "combined_with_small/llama_3.3_70b/fixed_seed/101/random"
    "combined_with_small/llama_3.3_70b/fixed_seed/101/ultrafeedback"
    "combined_with_small/llama_3.3_70b/fixed_seed/102/random"
    "combined_with_small/llama_3.3_70b/fixed_seed/102/ultrafeedback"
    "combined_with_small/llama_3.3_70b/fixed_seed/103/random"
    "combined_with_small/llama_3.3_70b/fixed_seed/103/ultrafeedback"
    "combined_with_small/llama_3.3_70b/fixed_seed/104/random"
    "combined_with_small/llama_3.3_70b/fixed_seed/104/ultrafeedback"
    "combined_with_small/qwen_3_235b/max_min"
    "combined_with_small/qwen_3_235b/fixed_seed/100/random"
    "combined_with_small/qwen_3_235b/fixed_seed/100/ultrafeedback"
    "combined_with_small/qwen_3_235b/fixed_seed/101/random"
    "combined_with_small/qwen_3_235b/fixed_seed/101/ultrafeedback"
    "combined_with_small/qwen_3_235b/fixed_seed/102/random"
    "combined_with_small/qwen_3_235b/fixed_seed/102/ultrafeedback"
    "combined_with_small/qwen_3_235b/fixed_seed/103/random"
    "combined_with_small/qwen_3_235b/fixed_seed/103/ultrafeedback"
    "combined_with_small/qwen_3_235b/fixed_seed/104/random"
    "combined_with_small/qwen_3_235b/fixed_seed/104/ultrafeedback"
    "combined_with_small/llama_8b_skywork_rm/max_min"
    "combined_with_small/llama_8b_skywork_rm/fixed_seed/100/random"
    "combined_with_small/llama_8b_skywork_rm/fixed_seed/100/ultrafeedback"
    "combined_with_small/llama_8b_skywork_rm/fixed_seed/101/random"
    "combined_with_small/llama_8b_skywork_rm/fixed_seed/101/ultrafeedback"
    "combined_with_small/llama_8b_skywork_rm/fixed_seed/102/random"
    "combined_with_small/llama_8b_skywork_rm/fixed_seed/102/ultrafeedback"
    "combined_with_small/llama_8b_skywork_rm/fixed_seed/103/random"
    "combined_with_small/llama_8b_skywork_rm/fixed_seed/103/ultrafeedback"
    "combined_with_small/llama_8b_skywork_rm/fixed_seed/104/random"
    "combined_with_small/llama_8b_skywork_rm/fixed_seed/104/ultrafeedback"
)

# SUBMIT JOBS
for run_suffix in "${RUNS[@]}"; do
    # Dynamically construct the full paths for dataset and output
    DATASET_PATH="${DATASETS_BASE_DIR}/${run_suffix}"
    OUTPUT_DIR="${MODELS_BASE_DIR}/${run_suffix}"

    JOB_NAME="rm-$(echo "${run_suffix}" | tr '/' '-')"
    
    echo "Submitting job for run: ${run_suffix}"
    echo "  -> Job Name: ${JOB_NAME}"
    echo "  -> Output Dir: ${OUTPUT_DIR}"
    echo "  -> Dataset: ${DATASET_PATH}"


    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH -D .
#SBATCH -A a-infra01-1
#SBATCH --output=./logs/reward/${JOB_NAME}_%j.out
#SBATCH --error=./logs/reward/${JOB_NAME}_%j.err
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=288         # number of cores per tasks
#SBATCH --time=12:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --environment=activeuf_dev      # using compressed docker image as an environment

export GPUS_PER_NODE=4
export HF_HOME="${HF_HOME}"

export WANDB_ENTITY=ActiveUF
export WANDB_PROJECT=RM_Training
export ACCELERATE_DIR="/accelerate"

export ACCELERATE_CONFIG="${ACCELERATE_CONFIG}"
export REWARD_CONFIG="${REWARD_CONFIG}"
export PYTHON_FILE="${PYTHON_FILE}"

######################
#### Set network #####
######################
head_node_ip=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --config_file=$ACCELERATE_CONFIG \
    --num_processes \$((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines \$SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip \$head_node_ip \
    --main_process_port 29500 \
    "

export ACCELERATE_DIR="\${ACCELERATE_DIR:-/accelerate}"

echo "Using ACCELERATE_DIR=\${ACCELERATE_DIR}"

export SCRIPT_ARGS=" \
    --output_dir $OUTPUT_DIR \
    --reward_config $REWARD_CONFIG \
    --dataset_path $DATASET_PATH \
    "

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="\$LAUNCHER \$PYTHON_FILE \$SCRIPT_ARGS"

START=\$(date +%s)

cd ${PROJECT_BASE_DIR}

echo \$CMD
srun \$CMD

pip install resources/reward-bench

accelerate launch --config_file=${PROJECT_BASE_DIR}/configs/accelerate/multi_node.yaml resources/reward-bench/scripts/run_v2.py --do_not_save --model="$OUTPUT_DIR"

END=\$(date +%s)
DURATION=\$(( END - START ))

echo "Job ended at: \$(date)"
echo "Total execution time: \$DURATION seconds"
EOF
done

echo -e "\nSuccessfully submitted ${#RUNS[@]} job(s)."