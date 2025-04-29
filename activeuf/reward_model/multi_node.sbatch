#!/bin/bash

#SBATCH --job-name=multinode
#SBATCH -D .
#SBATCH --output=multinode/O-%x.%j
#SBATCH --error=multinode/E-%x.%j
#SBATCH --nodes=4                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=288         # number of cores per tasks
#SBATCH --time=01:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --environment=activeuf      # using compressed docker image as an environment

export GPUS_PER_NODE=4
export HF_HOME=$SCRATCH/huggingface
######################

echo $ACCELERATE_DIR

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --config_file=$SCRATCH/reward-model-training2/multi_node.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    "

export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"
export PYTHON_FILE="${SCRATCH}/reward-model-training2/reward-trainer3.py"
export SCRIPT_ARGS=" \
    --output_dir ${SCRATCH}/reward-model-training2/trainedModels/NewModelTrainedMultiNodeWow3 \
    "
        
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $SCRIPT_ARGS" 

START=$(date +%s)

cd $SCRATCH/reward-model-training2/

srun $CMD

END=$(date +%s)
DURATION=$(( END - START ))

echo "Job ended at: $(date)"
echo "Total execution time: $DURATION seconds"