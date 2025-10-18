#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --account=a-infra01-1
#SBATCH --exclusive
#SBATCH --cpus-per-task=288
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --container-writable
#SBATCH --job-name=Qwen3-235B-A22B
#SBATCH --output=./logs/completions/Qwen3-235B-A22B/%j.out
#SBATCH --environment=activeuf_dev
# DISABLED SBATCH --exclude=nid006438,nid006439,nid006440,nid006441,nid006442,nid006443,nid006444,nid006445,nid006446,nid006447,nid006448,nid006449,nid006450,nid006451,nid006461,nid006462,nid006868,nid006476,nid005557,nid006455,nid007122,nid007119,nid006513,nid005813,nid006454,nid006452,nid006457,nid005230,nid005248

export RAY_CGRAPH_get_timeout=300
export VLLM_SERVER_CONCURRENCY_LIMIT=100
num_nodes_per_instance=2

export HF_HOME=/iopsstor/scratch/cscs/smarian/hf_cache
export WANDB_DIR=/iopsstor/scratch/cscs/smarian/wandb
export TRANSFORMERS_CACHE=/iopsstor/scratch/cscs/smarian/.cache/transformers
export HF_DATASETS_CACHE=/iopsstor/scratch/cscs/smarian/.cache/datasets
export TORCH_HOME=/iopsstor/scratch/cscs/smarian/.cache/torch
export XDG_CACHE_HOME=/iopsstor/scratch/cscs/smarian/.cache/.cache
export TORCH_EXTENSIONS_DIR=$XDG_CACHE_HOME/torch_extensions

# Getting the node names and assigning a head node
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

echo "nodes: ${nodes}"
echo "nodes_array: ${nodes_array[*]}"

head_node=${nodes_array[0]}
head_node_ip=$(srun --overlap --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"

export head_node_ip="$head_node_ip"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# If we detect a space character in the head node IP, we'll convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

# Start head node
port=6382
head_address=$head_node_ip:$port
export RAY_ADDRESS="$head_address"
export VLLM_HOST_IP="$head_node_ip"

echo "Head Address (set as RAY_ADDRESS): $head_address"
echo "Head VLLM_HOST_IP (set for driver): $VLLM_HOST_IP"

echo "Starting HEAD at $head_node at IP $head_node_ip"
ray start --head \
          --node-ip-address=$head_node_ip \
          --port=$port \
          --num-cpus=${SLURM_CPUS_PER_TASK} \
          --num-gpus=4  \
          --resources="{\"node:$head_node_ip\": 1}" \
          --block & 
sleep 10  

# Start workers
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node=${nodes_array[$i]}
    node_ip=$(srun --nodes=1 --overlap --ntasks=1 -w "$node" hostname --ip-address)
    echo "Starting WORKER $i at $node with IP $node_ip"

    srun --nodes=1 \
       --ntasks=1 \
       --overlap \
       -w "$node" \
       env VLLM_HOST_IP="$node_ip" CUDA_VISIBLE_DEVICES="0,1,2,3" \
       ray start --address $head_address \
                 --node-ip-address="$node_ip" \
                 --num-cpus ${SLURM_CPUS_PER_TASK} \
                 --num-gpus 4 \
                 --block &
    sleep 5
done
sleep 10

ray status

python -u -m activeuf.generate_completions \
  --dataset_path /iopsstor/scratch/cscs/smarian/datasets/0_raw_datasets/Skywork-Reward-Preference-80K-v0.2/train \
  --model_name Qwen/Qwen3-235B-A22B \
  --model_class vllm_server \
  --output_path /iopsstor/scratch/cscs/smarian/datasets/1_partial_completions/skywork/Qwen3-235B-A22B_0 \
  --seed 21 \
  --num_nodes $num_nodes_per_instance \
  --data_parallel_size $((SLURM_JOB_NUM_NODES / num_nodes_per_instance)) \
  --num_chunks 3 \
  --chunk_index 0
