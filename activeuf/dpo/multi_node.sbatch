#!/bin/bash

#SBATCH --job-name=multinode
#SBATCH -D .
#SBATCH -A a-infra01-1
#SBATCH --output=/iopsstor/scratch/cscs/jessica/logs/dpo/multinode/O-%x.%j
#SBATCH --error=/iopsstor/scratch/cscs/jessica/logs/dpo/multinode/E-%x.%j
#SBATCH --nodes=4                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=288         # number of cores per tasks
#SBATCH --time=01:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --environment=activeuf_dev  # using compressed docker image as an environment
#SBATCH --exclude=nid006438,nid006439,nid006440,nid006441,nid006442,nid006443,nid006444,nid006445,nid006446,nid006447,nid006448,nid006449,nid006450,nid006451,nid006461,nid006462,nid006868
#SBATCH --job-name=dummy_dpo_run

export GPUS_PER_NODE=4
export WANDB_ENTITY=ActiveUF
export WANDB_PROJECT=DPO
######################

echo $ACCELERATE_DIR

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --config_file=$SCRATCH/ActiveUltraFeedback/activeuf/dpo/multi_node.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    "

export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"
export PYTHON_FILE="-m activeuf.dpo.training"
export SCRIPT_ARGS=" \
    --config_path $SCRATCH/ActiveUltraFeedback/activeuf/dpo/training.yaml \
    --debug
    "
        
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $SCRIPT_ARGS" 

START=$(date +%s)

cd $SCRATCH/ActiveUltraFeedback/
echo "$(pwd)"
srun $CMD

END=$(date +%s)
DURATION=$(( END - START ))

echo "Job ended at: $(date)"
echo "Total execution time: $DURATION seconds"