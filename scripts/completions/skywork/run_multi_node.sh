#!/bin/bash

CACHE_DIR="$SCRATCH/cache"
DATASETS_DIR="$SCRATCH/datasets"

# Run configs: 
# Format: MODEL:NODES:CHUNK_INDEX:NUM_CHUNKS:SEED:CONCURRENCY:MAX_MODEL_LEN
#   MODEL: model path
#   NODES: number of nodes
#   CHUNK_INDEX: index of this chunk (0-indexed)
#   NUM_CHUNKS: total number of chunks the dataset is split into
#   SEED: seed for this specific chunk
#   CONCURRENCY: concurrency limit
#   MAX_MODEL_LEN: maximum model length (leave empty if not needed)
CONFIGS=(
    # Qwen/Qwen3-235B-A22B (2 nodes, 3 chunks)
    "Qwen/Qwen3-235B-A22B:2:0:3:21:100:"
    "Qwen/Qwen3-235B-A22B:2:1:3:22:100:"
    "Qwen/Qwen3-235B-A22B:2:2:3:23:100:"
    
    # allenai/Llama-3.1-Tulu-3-405B (4 nodes, 2 chunks)
    "allenai/Llama-3.1-Tulu-3-405B:4:0:2:24:100:"
    "allenai/Llama-3.1-Tulu-3-405B:4:1:2:25:100:"
    
    # nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 (4 nodes, 4 chunks)
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:0:4:19:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:1:4:20:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:2:4:21:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:3:4:22:50:15000"
    
    # deepseek-ai/DeepSeek-V3 (6 nodes, 4 chunks)
    "deepseek-ai/DeepSeek-V3:6:0:4:18:100:15000"
    "deepseek-ai/DeepSeek-V3:6:1:4:19:100:15000"
    "deepseek-ai/DeepSeek-V3:6:2:4:20:100:15000"
    "deepseek-ai/DeepSeek-V3:6:3:4:21:100:15000"
)

for config in "${CONFIGS[@]}"; do
    # Split the config
    IFS=':' read -r MODEL NODES CHUNK_IDX NUM_CHUNKS SEED CONCURRENCY MAX_MODEL_LEN <<< "$config"
    MODEL_NAME="${MODEL##*/}"
    
    JOB_NAME="${MODEL_NAME}_${CHUNK_IDX}"
    OUTPUT_PATH="${DATASETS_DIR}/1_partial_completions/skywork/${JOB_NAME}"
    
    echo "Submitting job for model: $MODEL_NAME, chunk: $CHUNK_IDX/$NUM_CHUNKS, nodes: $NODES, seed: $SEED"
    
    # Build the max_model_len argument if specified
    MAX_MODEL_LEN_ARG=""
    if [ -n "$MAX_MODEL_LEN" ]; then
        MAX_MODEL_LEN_ARG="  --max_model_len $MAX_MODEL_LEN \\\\"
    fi
    
    sbatch <<EOF
#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --account=a-infra01-1
#SBATCH --exclusive
#SBATCH --cpus-per-task=288
#SBATCH --nodes=$NODES
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --container-writable
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=./logs/completions/skywork/$MODEL_NAME/%j.out
#SBATCH --environment=activeuf_dev

export RAY_CGRAPH_get_timeout=300
export VLLM_SERVER_CONCURRENCY_LIMIT=$CONCURRENCY
num_nodes_per_instance=$NODES

export HF_HOME=$CACHE_DIR/hf_cache
export WANDB_DIR=$CACHE_DIR/wandb
export TRANSFORMERS_CACHE=$CACHE_DIR/transformers
export HF_DATASETS_CACHE=$CACHE_DIR/datasets
export TORCH_HOME=$CACHE_DIR/torch
export XDG_CACHE_HOME=$CACHE_DIR
export TORCH_EXTENSIONS_DIR=\$XDG_CACHE_HOME/torch_extensions

# Getting the node names and assigning a head node
nodes=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST")
nodes_array=(\$nodes)

echo "nodes: \${nodes}"
echo "nodes_array: \${nodes_array[*]}"

head_node=\${nodes_array[0]}
head_node_ip=\$(srun --overlap --nodes=1 --ntasks=1 -w "\$head_node" hostname --ip-address)

echo "Head node: \$head_node"
echo "Head node IP: \$head_node_ip"

export head_node_ip="\$head_node_ip"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# If we detect a space character in the head node IP, we'll convert it to an ipv4 address
if [[ "\$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"\$head_node_ip"
if [[ \${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=\${ADDR[1]}
else
  head_node_ip=\${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as \$head_node_ip"
fi

# Start head node
port=6382
head_address=\$head_node_ip:\$port
export RAY_ADDRESS="\$head_address"
export VLLM_HOST_IP="\$head_node_ip"

echo "Head Address (set as RAY_ADDRESS): \$head_address"
echo "Head VLLM_HOST_IP (set for driver): \$VLLM_HOST_IP"

echo "Starting HEAD at \$head_node at IP \$head_node_ip"
ray start --head \\
          --node-ip-address=\$head_node_ip \\
          --port=\$port \\
          --num-cpus=\${SLURM_CPUS_PER_TASK} \\
          --num-gpus=4  \\
          --resources="{\"node:\$head_node_ip\": 1}" \\
          --block & 
sleep 10  

# Start workers
worker_num=\$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node=\${nodes_array[\$i]}
    node_ip=\$(srun --nodes=1 --overlap --ntasks=1 -w "\$node" hostname --ip-address)
    echo "Starting WORKER \$i at \$node with IP \$node_ip"

    srun --nodes=1 \\
       --ntasks=1 \\
       --overlap \\
       -w "\$node" \\
       env VLLM_HOST_IP="\$node_ip" CUDA_VISIBLE_DEVICES="0,1,2,3" \\
       ray start --address \$head_address \\
                 --node-ip-address="\$node_ip" \\
                 --num-cpus \${SLURM_CPUS_PER_TASK} \\
                 --num-gpus 4 \\
                 --block &
    sleep 5
done
sleep 10

ray status

python -u -m activeuf.completions.generate_completions \
  --dataset_path ${DATASETS_DIR}/0_raw_datasets/llama-3.1-tulu-3-8b-preference-mixture/ \
  --model_name $MODEL \
  --model_class vllm_server \
  --output_path $OUTPUT_PATH \
  --seed $SEED \
  --num_nodes \$num_nodes_per_instance \
  --data_parallel_size \$((SLURM_JOB_NUM_NODES / num_nodes_per_instance)) \
$MAX_MODEL_LEN_ARG
  --num_chunks $NUM_CHUNKS \
  --chunk_index $CHUNK_IDX
EOF
done

