#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=test_ray
#SBATCH --cpus-per-task=288
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --environment=activeuf_dev
#SBATCH --account=a-infra01-1
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --output=./logs/ray/ray_test_%j.out

echo -e "========================ray_test.sh============================="
cat ./scripts/start_ray_cluster.sh
echo -e "\n==============================================================\n\n\n"


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

# Run the command you want to use Ray for here
vllm serve Qwen/Qwen3-235B-A22B \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --swap-space 1 \
    --trust-remote-code \
    --dtype auto \
    --download-dir /iopsstor/scratch/cscs/smarian/hf_cache \
    --port 8000 &

# Wait for the server to start (there is probably a better way to do this)
sleep 600

curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-235B-A22B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'