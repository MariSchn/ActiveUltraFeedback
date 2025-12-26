#!/bin/bash

CACHE_DIR="$SCRATCH/cache"
DATASETS_DIR="$SCRATCH/datasets"

# Optional: Set this to a job ID to make all submitted jobs wait for it to finish
# Example: WAIT_FOR_JOB=1234567
WAIT_FOR_JOB="1256136"

# Run configs: 
# Format: MODEL:NODES:CHUNK_INDEX:NUM_CHUNKS:SEED:CONCURRENCY:MAX_MODEL_LEN
#   MODEL: model path
#   NODES: number of nodes
#   CHUNK_INDEX: index of this chunk (0-indexed), use -1 if not chunked
#   NUM_CHUNKS: total number of chunks the dataset is split into, use 0 if not chunked
#   SEED: seed for this specific job
#   CONCURRENCY: concurrency limit
#   MAX_MODEL_LEN: maximum model length (leave empty if not needed)
CONFIGS=(
    # Qwen/Qwen3-235B-A22B (2 nodes, 8 chunks)
    "Qwen/Qwen3-235B-A22B:2:0:8:2:100:"
    "Qwen/Qwen3-235B-A22B:2:1:8:2:100:"
    "Qwen/Qwen3-235B-A22B:2:2:8:2:100:"
    "Qwen/Qwen3-235B-A22B:2:3:8:2:100:"
    "Qwen/Qwen3-235B-A22B:2:4:8:2:100:"
    "Qwen/Qwen3-235B-A22B:2:5:8:2:100:"
    "Qwen/Qwen3-235B-A22B:2:6:8:2:100:"
    "Qwen/Qwen3-235B-A22B:2:7:8:2:100:"
    
    # allenai/Llama-3.1-Tulu-3-405B (4 nodes, 5 chunks)
    "allenai/Llama-3.1-Tulu-3-405B:4:0:10:19310:100:"
    "allenai/Llama-3.1-Tulu-3-405B:4:1:10:19310:100:"
    "allenai/Llama-3.1-Tulu-3-405B:4:2:10:19310:100:"
    "allenai/Llama-3.1-Tulu-3-405B:4:3:10:19310:100:"
    "allenai/Llama-3.1-Tulu-3-405B:4:4:10:19310:100:"
    "allenai/Llama-3.1-Tulu-3-405B:4:5:10:19310:100:"
    "allenai/Llama-3.1-Tulu-3-405B:4:6:10:19310:100:"
    "allenai/Llama-3.1-Tulu-3-405B:4:7:10:19310:100:"
    "allenai/Llama-3.1-Tulu-3-405B:4:8:10:19310:100:"
    "allenai/Llama-3.1-Tulu-3-405B:4:9:10:19310:100:"
    
    # nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 (4 nodes, 8 chunks)
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:0:16:7122:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:1:16:7123:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:2:16:7123:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:3:16:7123:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:4:16:7123:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:5:16:7123:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:6:16:7123:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:7:16:7123:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:8:16:7123:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:9:16:7123:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:10:16:7123:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:11:16:7123:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:12:16:7123:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:13:16:7123:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:14:16:7123:50:15000"
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1:4:15:16:7123:50:15000"
    
    # deepseek-ai/DeepSeek-V3 (6 nodes, 10 chunks)
    "deepseek-ai/DeepSeek-V3:6:0:10:9862:100:15000"
    "deepseek-ai/DeepSeek-V3:6:1:10:9862:100:15000"
    "deepseek-ai/DeepSeek-V3:6:2:10:9862:100:15000"
    "deepseek-ai/DeepSeek-V3:6:3:10:9862:100:15000"
    "deepseek-ai/DeepSeek-V3:6:4:10:9862:100:15000"
    "deepseek-ai/DeepSeek-V3:6:5:10:9862:100:15000"
    "deepseek-ai/DeepSeek-V3:6:6:10:9862:100:15000"
    "deepseek-ai/DeepSeek-V3:6:7:10:9862:100:15000"
    "deepseek-ai/DeepSeek-V3:6:8:10:9862:100:15000"
    "deepseek-ai/DeepSeek-V3:6:9:10:9862:100:15000"
)

for config in "${CONFIGS[@]}"; do
    # Split the config
    IFS=':' read -r MODEL NODES CHUNK_IDX NUM_CHUNKS SEED CONCURRENCY MAX_MODEL_LEN <<< "$config"
    MODEL_NAME="${MODEL##*/}"
    
    # Determine job name, output path, and log path based on whether chunking is used
    if [ "$CHUNK_IDX" -eq -1 ]; then
        JOB_NAME="${MODEL_NAME}"
        OUTPUT_PATH="${DATASETS_DIR}/2_full_completions/tulu_3/${MODEL_NAME}"
        LOG_PATH="./logs/completions/tulu_3/${MODEL_NAME}/%j.out"
    else
        JOB_NAME="${MODEL_NAME}_${CHUNK_IDX}"
        OUTPUT_PATH="${DATASETS_DIR}/1_partial_completions/tulu_3/${JOB_NAME}"
        LOG_PATH="./logs/completions/tulu_3/${MODEL_NAME}/chunk_${CHUNK_IDX}_%j.out"
    fi
    
    echo "Submitting job for model: $MODEL_NAME (chunk: $CHUNK_IDX/$NUM_CHUNKS), nodes: $NODES, seed: $SEED"
    
    # Build optional arguments
    OPTIONAL_ARGS=""
    if [ -n "$MAX_MODEL_LEN" ]; then
        OPTIONAL_ARGS="$OPTIONAL_ARGS --max_model_len $MAX_MODEL_LEN"
    fi
    if [ "$CHUNK_IDX" -ne -1 ]; then
        OPTIONAL_ARGS="$OPTIONAL_ARGS --num_chunks $NUM_CHUNKS --chunk_index $CHUNK_IDX"
    fi
    
    # Build dependency argument if WAIT_FOR_JOB is set
    DEPENDENCY_ARG=""
    if [ -n "$WAIT_FOR_JOB" ]; then
        DEPENDENCY_ARG="--dependency=afterany:$WAIT_FOR_JOB"
        echo "  -> Will wait for job $WAIT_FOR_JOB to finish first"
    fi
    
    sbatch $DEPENDENCY_ARG <<EOF
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
#SBATCH --output=$LOG_PATH
#SBATCH --environment=activeuf_10_25

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
      bash -c "export VLLM_HOST_IP=\$node_ip CUDA_VISIBLE_DEVICES=0,1,2,3; \
              ray start --address \$head_address \
                        --node-ip-address=\$node_ip \
                        --num-cpus \${SLURM_CPUS_PER_TASK} \
                        --num-gpus 4 \
                        --block" &
    sleep 5
done
sleep 10

ray status

python -u -m activeuf.completions.generate_completions \\
  --dataset_path $DATASETS_DIR/0_raw_datasets/llama-3.1-tulu-3-8b-preference-mixture/ \\
  --model_name $MODEL \\
  --model_class vllm_server \\
  --output_path $OUTPUT_PATH \\
  --seed $SEED \\
  --num_nodes \$num_nodes_per_instance \\
  --data_parallel_size \$((SLURM_JOB_NUM_NODES / num_nodes_per_instance)) \\
  $OPTIONAL_ARGS
EOF
done

