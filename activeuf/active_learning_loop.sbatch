#!/bin/bash

#SBATCH --job-name=multinodeloop
#SBATCH -D .
#SBATCH -A a-a145
#SBATCH --output=multinodeloop/O-%x.%j
#SBATCH --error=multinodeloop/E-%x.%j
#SBATCH --nodes=8                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=288         # number of cores per tasks
#SBATCH --time=04:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --environment=activeuf      # using compressed docker image as an environment

export GPUS_PER_NODE=4
export HF_HOME=$SCRATCH/huggingface
export HF_TOKEN=<Huggingface Token>
######################

echo $ACCELERATE_DIR

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --config_file=$SCRATCH/ActiveUltraFeedback/activeuf/reward_model/multi_node.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    -m
    "

export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"
export PYTHON_FILE="activeuf.active_learning_loop"
export SCRIPT_ARGS=" \
   --completions_dataset_path ${SCRATCH}/datasets/ultrafeedback_annotated/ \
   --output_path=$SCRATCH/datasets/active_davit_final/ \
   --logs_path=$SCRATCH/logs_final/ \
   --args_path=$SCRATCH/models_enn \
   --acquisition_config=/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/activeuf/acquisition_function/configs.yaml \
   "
        
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $SCRIPT_ARGS" 

export DS_ACCELERATOR_CACHE_DIR="$SCRATCH/tmp/deepspeed_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$SCRATCH/tmp/hf_cache_$SLURM_JOB_ID"
mkdir -p "$DS_ACCELERATOR_CACHE_DIR" "$TRANSFORMERS_CACHE"

START=$(date +%s)

cd $SCRATCH/ActiveUltraFeedback/
export PYTHONPATH="$SCRATCH/ActiveUltraFeedback:$PYTHONPATH"
pip install --user git+https://github.com/Florian-toll/rewarduq.git

srun --chdir=$SCRATCH/ActiveUltraFeedback $CMD

END=$(date +%s)
DURATION=$(( END - START ))

echo "Job ended at: $(date)"
echo "Total execution time: $DURATION seconds"