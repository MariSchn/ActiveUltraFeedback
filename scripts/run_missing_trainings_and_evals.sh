#!/bin/bash

# Configuration paths (adjust as needed)
export DPO_CONFIG="./configs/dpo_training.yaml"
export RM_CONFIG="./configs/rm_training.yaml"
export ACCELERATE_CONFIG="./configs/accelerate/multi_node.yaml"

export WANDB_ENTITY=ActiveUF
export WANDB_DIR="${SCRATCH}/cache/wandb"
export BASE_RM_OUTPUT_DIR="${SCRATCH}/models/reward_models"
export BASE_DPO_OUTPUT_DIR="${SCRATCH}/models/dpo"
export HF_HOME="${SCRATCH}/cache/hf_cache"
export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"

export GPUS_PER_NODE=4

# Parse args
DATASETS_BASE_DIR=""
RM_MODEL_BASE_DIR=""
DPO_MODEL_BASE_DIR=""
SWEEP_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_base_dir)
            DATASETS_BASE_DIR="$2"
            shift 2
            ;;
        --rm_base_dir)
            RM_MODEL_BASE_DIR="$2"
            shift 2
            ;;
        --dpo_base_dir)
            DPO_MODEL_BASE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --dataset_base_dir <path> --rm_base_dir <path> --dpo_base_dir <path>"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATASETS_BASE_DIR" ]] || [[ -z "$RM_MODEL_BASE_DIR" ]] || [[ -z "$DPO_MODEL_BASE_DIR" ]]; then
    echo "Error: Please provide all required arguments."
    echo "Usage: $0 --dataset_base_dir <path> --rm_base_dir <path> --dpo_base_dir <path>"
    exit 1
fi

echo -e "====================================="
echo -e "===== LOOKING FOR LOOP DATASETS ====="
echo -e "=====================================\n"

# Check if loop dataset base directory exists
if [[ ! -d "$DATASETS_BASE_DIR" ]]; then
    echo "Error: Loop dataset base directory does not exist: $DATASETS_BASE_DIR"
    exit 1
fi

# Get all directories in DATASETS_BASE_DIR
datasets=()
for dir in "$DATASETS_BASE_DIR"/*; do
    if [[ -d "$dir" ]]; then
        datasets+=("$(basename "$dir")")
    fi
done

echo "Found ${#datasets[@]} datasets (directories) in $DATASETS_BASE_DIR"


echo -e "\n================================="
echo -e "===== CHECKING RM TRAININGS ====="
echo -e "=================================\n"

# Check for missing RM models
echo "--- Looking for missing reward models... ---"
missing_rm_dirs=()
for dir_name in "${datasets[@]}"; do
    config_file="${RM_MODEL_BASE_DIR}/${dir_name}/config.json"
    
    if [[ ! -f "$config_file" ]]; then
        echo "  Missing reward model for: $dir_name (no config.json at $config_file)"
        missing_rm_dirs+=("$dir_name")
    fi
done

if [[ ${#missing_rm_dirs[@]} -eq 0 ]]; then
    echo "All RM models are present!"
else
    echo "Found ${#missing_rm_dirs[@]} missing RM models"
    echo ""
    echo "--- Launching RM training jobs ---"
    for dir_name in "${missing_rm_dirs[@]}"; do
        dataset_path="$DATASETS_BASE_DIR/$dir_name"
        rm_output_path="${RM_MODEL_BASE_DIR}/${dir_name}"

        echo "Processing: $dir_name"
        echo "  Dataset path: $dataset_path"
        echo "  RM output path: $rm_output_path"
        
        # Create output directory for logs
        mkdir -p "${rm_output_path}"
        
        # Submit RM training job
        echo "  Submitting RM training job..."
        sbatch --job-name="rm_${dir_name}" \
               -A "a-infra01-1" \
               --output="${rm_output_path}/training_%j.log" \
               --nodes=2 \
               --time=12:00:00 \
               --partition=normal \
               --wrap="
                   export WANDB_ENTITY=${WANDB_ENTITY}
                   export WANDB_PROJECT=RM
                   export HF_HOME=${HF_HOME}
                   export WANDB_DIR=${WANDB_DIR}
                   
                   export RM_NUM_PROCESSES=\$(expr \$SLURM_NNODES \* $GPUS_PER_NODE)
                   export RM_NODELIST=(\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\"))
                   export RM_HEAD_NODE=\${RM_NODELIST[0]}
                   export RM_HEAD_NODE_IP=\$(srun --nodes=1 --ntasks=1 -w \"\$RM_HEAD_NODE\" hostname -i)
                   export RM_HEAD_PROCESS_PORT=29500

                   RM_TRAIN_LAUNCHER=\"accelerate launch \
                       --config_file ${ACCELERATE_CONFIG} \
                       --num_processes \$RM_NUM_PROCESSES \
                       --num_machines \$SLURM_NNODES \
                       --rdzv_backend c10d \
                       --main_process_ip \$RM_HEAD_NODE_IP \
                       --main_process_port \$RM_HEAD_PROCESS_PORT\"
                   RM_TRAIN_ARGS=\"\
                       --output_dir ${rm_output_path} \
                       --reward_config ${RM_CONFIG} \
                       --dataset_path ${dataset_path}\"
                   RM_TRAIN_CMD=\"\$RM_TRAIN_LAUNCHER ./activeuf/reward_model/training.py \$RM_TRAIN_ARGS\"
                   
                   echo \"Running command: \$RM_TRAIN_CMD\"
                   srun --environment=activeuf_dev \$RM_TRAIN_CMD
                   
                   echo \"Training complete. Starting RM evaluation...\"
                   srun --environment=activeuf_dev --nodes=1 --ntasks=1 \
                        bash ./activeuf/reward_model/reward_bench_2.sh --model ${rm_output_path}
               "
        
        echo "  Job submitted for $dir_name"
        echo ""
    done
fi

echo -e "\n=================================="
echo -e "===== CHECKING DPO TRAININGS ====="
echo -e "==================================\n"

# Check for missing DPO models
echo "--- Looking for missing DPO models... ---"
missing_dpo_dirs=()
for dir_name in "${datasets[@]}"; do
    config_file="${DPO_MODEL_BASE_DIR}/${dir_name}/config.json"
    
    if [[ ! -f "$config_file" ]]; then
        echo "  Missing DPO model for: $dir_name (no config.json at $config_file)"
        missing_dpo_dirs+=("$dir_name")
    fi
done

if [[ ${#missing_dpo_dirs[@]} -eq 0 ]]; then
    echo "All DPO models are present!"
else
    echo "Found ${#missing_dpo_dirs[@]} missing DPO models"
    echo ""
    echo "--- Launching DPO training jobs ---"
    for dir_name in "${missing_dpo_dirs[@]}"; do
        dataset_path="$DATASETS_BASE_DIR/$dir_name"
        dpo_output_path="${DPO_MODEL_BASE_DIR}/${dir_name}"

        echo "Processing: $dir_name"
        echo "  Dataset path: $dataset_path"
        echo "  DPO output path: $dpo_output_path"
        
        # Create output directory for logs
        mkdir -p "${dpo_output_path}"
        
        # Submit DPO training job
        echo "  Submitting DPO training job..."
        sbatch --job-name="dpo_${dir_name}" \
               -A "a-infra01-1" \
               --output="${dpo_output_path}/training_%j.log" \
               --nodes=2 \
               --time=12:00:00 \
               --partition=normal \
               --wrap="
                   export WANDB_ENTITY=${WANDB_ENTITY}
                   export WANDB_PROJECT=DPO
                   export HF_HOME=${HF_HOME}
                   export WANDB_DIR=${WANDB_DIR}
                   
                   export DPO_NUM_PROCESSES=\$(expr \$SLURM_NNODES \* $GPUS_PER_NODE)
                   export DPO_NODELIST=(\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\"))
                   export DPO_HEAD_NODE=\${DPO_NODELIST[0]}
                   export DPO_HEAD_NODE_IP=\$(srun --nodes=1 --ntasks=1 -w \"\$DPO_HEAD_NODE\" hostname -i)
                   export DPO_HEAD_PROCESS_PORT=29500
                   
                   DPO_TRAIN_LAUNCHER=\"accelerate launch \
                       --config_file ${ACCELERATE_CONFIG} \
                       --num_processes \$DPO_NUM_PROCESSES \
                       --num_machines \$SLURM_NNODES \
                       --rdzv_backend c10d \
                       --main_process_ip \$DPO_HEAD_NODE_IP \
                       --main_process_port \$DPO_HEAD_PROCESS_PORT\"
                   DPO_TRAIN_ARGS=\"\
                       --config_path ${DPO_CONFIG} \
                       --slurm_job_id \$SLURM_JOB_ID \
                       --dataset_path ${dataset_path} \
                       --output_dir ${dpo_output_path}\"
                   DPO_TRAIN_CMD=\"\$DPO_TRAIN_LAUNCHER -m activeuf.dpo.training \$DPO_TRAIN_ARGS\"
                   
                   echo \"Running command: \$DPO_TRAIN_CMD\"
                   srun --environment=activeuf_dev \$DPO_TRAIN_CMD
                   
                   echo \"Training complete. Starting DPO evaluation...\"
                   
                   # Setup DPO evaluation environment
                   export VLLM_WORKER_MULTIPROC_METHOD=spawn
                   export PROJECT_ROOT_AT=\$SCRATCH/projects/ActiveUltraFeedback/resources/olmes
                   export PROJECT_NAME=olmes
                   export PACKAGE_NAME=oe_eval
                   export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
                   export WANDB_API_KEY_FILE_AT=\$HOME/.wandb-api-key
                   export SKIP_INSTALL_PROJECT=1
                   export SHARED=/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared
                   export OMP_NUM_THREADS=1
                   export TOKENIZERS_PARALLELISM=false
                   export CONTAINER_IMAGES=/capstor/store/cscs/swissai/infra01/container-images
                   unset SSL_CERT_FILE
                   
                   # Create results directories
                   mkdir -p ${dpo_output_path}/results/gsm8k_tulu \
                            ${dpo_output_path}/results/minerva_math_tulu \
                            ${dpo_output_path}/results/ifeval_tulu \
                            ${dpo_output_path}/results/truthfulqa_tulu
                   
                   CONTAINER_ARGS=\"--container-image=\$CONTAINER_IMAGES/infra01+ismayilz+olmes+arm64-cuda-root-latest.sqsh \
                     --environment=\$PROJECT_ROOT_AT/installation/edf.toml \
                     --container-mounts=\$PROJECT_ROOT_AT,\$SCRATCH,/iopsstor/scratch/cscs/smoalla/projects/dpr/,\$SHARED,\$WANDB_API_KEY_FILE_AT,\$HOME/.gitconfig,\$HOME/.bashrc,\$HOME/.ssh \
                     --container-workdir=\$PROJECT_ROOT_AT \
                     --no-container-mount-home \
                     --no-container-remap-root \
                     --no-container-entrypoint \
                     --container-writable \
                     /opt/template-entrypoints/pre-entrypoint.sh\"
                   
                   DPO_EVAL_ARGS=\"--model=${dir_name} \
                     --model-wb-name=${dir_name} \
                     --model-type=vllm \
                     --batch-size=1 \
                     --model-args '{\\\"tensor_parallel_size\\\": 4, \\\"max_length\\\": 4096, \\\"add_bos_token\\\": false, \\\"model_path\\\": \\\"${dpo_output_path}\\\", \\\"trust_remote_code\\\": true}' \
                     --use-chat-format=True\"
                   
                   # Start all evaluations in parallel
                   echo \"Starting all DPO evaluations in parallel...\"
                   
                   srun --output=${dpo_output_path}/results/gsm8k_tulu/log.out \
                        --nodes=1 --ntasks=1 --gpus-per-task=4 \
                        \$CONTAINER_ARGS bash -c \"exec python3 -m oe_eval.launch --task=gsm8k::tulu --output-dir=${dpo_output_path}/results/gsm8k_tulu \$DPO_EVAL_ARGS\" &
                   
                   srun --output=${dpo_output_path}/results/minerva_math_tulu/log.out \
                        --nodes=1 --ntasks=1 --gpus-per-task=4 \
                        \$CONTAINER_ARGS bash -c \"exec python3 -m oe_eval.launch --task=minerva_math::tulu --output-dir=${dpo_output_path}/results/minerva_math_tulu \$DPO_EVAL_ARGS\" &
                   
                   srun --output=${dpo_output_path}/results/ifeval_tulu/log.out \
                        --nodes=1 --ntasks=1 --gpus-per-task=4 \
                        \$CONTAINER_ARGS bash -c \"exec python3 -m oe_eval.launch --task=ifeval::tulu --output-dir=${dpo_output_path}/results/ifeval_tulu \$DPO_EVAL_ARGS\" &
                   
                   srun --output=${dpo_output_path}/results/truthfulqa_tulu/log.out \
                        --nodes=1 --ntasks=1 --gpus-per-task=4 \
                        \$CONTAINER_ARGS bash -c \"exec python3 -m oe_eval.launch --task=truthfulqa::tulu --output-dir=${dpo_output_path}/results/truthfulqa_tulu \$DPO_EVAL_ARGS\" &
                   
                   # Wait for all evaluations to complete
                   wait
                   echo \"All DPO evaluations complete.\"
               "
        
        echo "  Job submitted for $dir_name"
        echo ""
    done
fi

echo -e "\n==================="
echo -e "===== SUMMARY ====="
echo -e "===================\n"
echo "Total datasets: ${#datasets[@]}"
echo "RM jobs submitted: ${#missing_rm_dirs[@]}"
echo "DPO jobs submitted: ${#missing_dpo_dirs[@]}"
