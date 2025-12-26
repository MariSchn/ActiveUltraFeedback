#!/bin/bash
ANNOTATION_MODEL="Qwen/Qwen3-235B-A22B"
ANNOTATION_MODEL_SHORT=$(echo "$ANNOTATION_MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
BASE_DATASETS_DIR="/iopsstor/scratch/cscs/smarian/datasets"
DATASETS=(
  "tulu_3"
)

# Models to annotate
MODELS=(
"CohereLabs/c4ai-command-a-03-2025"
"deepseek-ai/DeepSeek-V3"
"google/gemma-3-1b-it"
"google/gemma-3-4b-it"
"google/gemma-3-12b-it"
"google/gemma-3-27b-it"
"nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"
"nvidia/Llama-3_3-Nemotron-Super-49B-v1"
"meta-llama/Llama-3.1-8B-Instruct"
"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
"allenai/Llama-3.1-Tulu-3-70B"
"allenai/Llama-3.1-Tulu-3-405B"
meta-llama/Llama-3.2-1B-Instruct
meta-llama/Llama-3.2-3B-Instruct
"meta-llama/Llama-3.3-70B-Instruct"
"mistralai/Mistral-Large-Instruct-2411"
"mistralai/Mistral-Small-24B-Instruct-2501"
"moonshotai/Moonlight-16B-A3B-Instruct"
"allenai/OLMo-2-0325-32B-Instruct"
"microsoft/phi-4"
"microsoft/Phi-4-mini-instruct"
"Qwen/Qwen2.5-0.5B-Instruct"
"Qwen/Qwen2.5-72B-Instruct"
"Qwen/Qwen3-0.6B"
"Qwen/Qwen3-1.7B"
"Qwen/Qwen3-14B"
"Qwen/Qwen3-30B-A3B"
"Qwen/Qwen3-32B"
"Qwen/Qwen3-235B-A22B"
"HuggingFaceTB/SmolLM2-1.7B-Instruct"
)

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
    for DATASET in "${DATASETS[@]}"; do
        sbatch <<EOF
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
#SBATCH --job-name=annotation_${MODEL_SHORT}
#SBATCH --output=./logs/annotation/${MODEL_SHORT}/%j.out
#SBATCH --environment=activeuf_dev
# DISABLED SBATCH --exclude=nid006438,nid006439,nid006440,nid006441,nid006442,nid006443,nid006444,nid006445,nid006446,nid006447,nid006448,nid006449,nid006450,nid006451,nid006461,nid006462,nid006868,nid006476,nid005557,nid006455,nid007122,nid007119,nid006513,nid005813,nid006454,nid006452,nid006457,nid005230,nid005248

export RAY_CGRAPH_get_timeout=300
export VLLM_SERVER_CONCURRENCY_LIMIT=100
num_nodes_per_instance=2

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

# If we detect a space character in the head node IP, we'll convert it to an ipv4 address. This step is optional.
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
ray start --head \
          --node-ip-address=\$head_node_ip \
          --port=\$port \
          --num-cpus=\${SLURM_CPUS_PER_TASK} \
          --num-gpus=4  \
          --resources="{\\"node:\$head_node_ip\\": 1}" \
          --block & 
sleep 10  

# Start workers
worker_num=\$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node=\${nodes_array[\$i]}
    node_ip=\$(srun --nodes=1 --overlap --ntasks=1 -w "\$node" hostname --ip-address)
    echo "Starting WORKER \$i at \$node with IP \$node_ip"

    srun --nodes=1 \
       --ntasks=1 \
       --overlap \
       -w "\$node" \
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

python -u -m activeuf.oracle.get_raw_annotations \
    --dataset_path ${BASE_DATASETS_DIR}/3_merged_completions/${DATASET} \
    --model_name="${ANNOTATION_MODEL}" \
    --max_tokens 24000 \
    --output_path ${BASE_DATASETS_DIR}/4_annotated_completions/${DATASET}/${MODEL_SHORT} \
    --model_class vllm_server \
    --temperature 0.0 \
    --top_p 0.1 \
    --num_nodes \$num_nodes_per_instance \
    --model_to_annotate "$MODEL" \
    --batch_size_to_annotate 5000
EOF

        echo "Submitted job for model: $MODEL on dataset: $DATASET using $ANNOTATION_MODEL"
        sleep 10

    done
done
