#!/bin/bash
# Script to annotate allenai/ultrafeedback_binarized_cleaned dataset
# using Qwen/Qwen3-235B-A22B as the judge model on a multi-node Ray cluster.
#
# This script submits a SLURM job that:
# 1. Starts a Ray cluster across 2 nodes (8 GPUs total)
# 2. Runs the annotation script for both chosen and rejected responses
#
# Usage:
#   ./run_annotations_binarized.sh
#
# Or to run with custom parameters:
#   ./run_annotations_binarized.sh --split train_prefs --batch_size 1000



# Default parameters
SPLIT="${SPLIT:-train_prefs}"
BATCH_SIZE="${BATCH_SIZE:-500}"
OUTPUT_BASE="${OUTPUT_BASE:-/iopsstor/scratch/cscs/dmelikidze/datasets/ultrafeedback_binarized_annotated}"
NUM_NODES="${NUM_NODES:-4}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
ENABLE_REASONING="${ENABLE_REASONING:-false}"
REASONING_MAX_TOKENS="${REASONING_MAX_TOKENS:-8192}"
DIRECT_OUTPUT="${DIRECT_OUTPUT:-false}"
DIRECT_OUTPUT_FIELD="${DIRECT_OUTPUT_FIELD:-}" # empty by default
BATCH_START="${BATCH_START:-}"
BATCH_END="${BATCH_END:-}"


# Parse command line arguments, including direct output options and batch intervals
while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --enable_reasoning)
            ENABLE_REASONING="true"
            shift 1
            ;;
        --reasoning_max_tokens)
            REASONING_MAX_TOKENS="$2"
            shift 2
            ;;
        --direct_output)
            DIRECT_OUTPUT="false"
            shift 1
            ;;
        --direct_output_field)
            DIRECT_OUTPUT_FIELD="$2"
            shift 2
            ;;
        --batch_start)
            BATCH_START="$2"
            shift 2
            ;;
        --batch_end)
            BATCH_END="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


# Add batch interval to output path for uniqueness if set
OUTPUT_PATH="${OUTPUT_BASE}/${SPLIT}"
if [[ -n "${BATCH_START}" ]] && [[ -n "${BATCH_END}" ]]; then
    OUTPUT_PATH="${OUTPUT_PATH}_batch_${BATCH_START}_${BATCH_END}"
fi

# Build reasoning flags for Python script
REASONING_FLAGS=""
if [[ "${ENABLE_REASONING}" == "true" ]]; then
    REASONING_FLAGS="--enable_reasoning --reasoning_max_tokens ${REASONING_MAX_TOKENS}"
fi

# --- MOVED UP: Build direct output flags ---
DIRECT_OUTPUT_FLAGS=""
if [[ "${DIRECT_OUTPUT}" == "true" ]]; then
    DIRECT_OUTPUT_FLAGS="--direct_output"
    if [[ -n "${DIRECT_OUTPUT_FIELD}" ]]; then
        DIRECT_OUTPUT_FLAGS+=" --direct_output_field ${DIRECT_OUTPUT_FIELD}"
    fi
fi

BATCH_INTERVAL_FLAGS=""
if [[ -n "${BATCH_START}" ]]; then
    BATCH_INTERVAL_FLAGS="--batch_start ${BATCH_START}"
fi
if [[ -n "${BATCH_END}" ]]; then
    BATCH_INTERVAL_FLAGS+=" --batch_end ${BATCH_END}"
fi

echo "=== Submitting annotation job ==="
echo "  Dataset: allenai/ultrafeedback_binarized_cleaned"
echo "  Split: ${SPLIT}"
echo "  Model: Qwen/Qwen3-235B-A22B"
echo "  Output: ${OUTPUT_PATH}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Nodes: ${NUM_NODES}"
echo "  Time limit: ${TIME_LIMIT}"
echo "  HF_HOME: ${HF_HOME}"
echo "  Enable Reasoning: ${ENABLE_REASONING}"
echo "  Reasoning Max Tokens: ${REASONING_MAX_TOKENS}"
echo "  Direct Output: ${DIRECT_OUTPUT}"
echo "  Direct Output Field: ${DIRECT_OUTPUT_FIELD}"
if [[ -n "${BATCH_START}" ]] && [[ -n "${BATCH_END}" ]]; then
    echo "  Batch Interval: ${BATCH_START} to ${BATCH_END}"
fi
echo "================================"

# Create log directory
mkdir -p ./logs/annotation/binarized

sbatch <<EOF
#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --account=a-infra01-1
#SBATCH --exclusive
#SBATCH --cpus-per-task=288
#SBATCH --nodes=${NUM_NODES}
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --partition=normal
#SBATCH --time=${TIME_LIMIT}
#SBATCH --container-writable
#SBATCH --job-name=annot_binarized
#SBATCH --output=./logs/annotation/binarized/%j.out
#SBATCH --environment=activeuf_dev

export RAY_CGRAPH_get_timeout=300
export VLLM_SERVER_CONCURRENCY_LIMIT=100
num_nodes_per_instance=${NUM_NODES}

# Getting the node names and assigning a head node
nodes=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST")
nodes_array=(\$nodes)

echo "========================================"
echo "Job ID: \$SLURM_JOB_ID"
echo "Nodes: \${nodes}"
echo "Nodes array: \${nodes_array[*]}"
echo "========================================"

head_node=\${nodes_array[0]}
head_node_ip=\$(srun --overlap --nodes=1 --ntasks=1 -w "\$head_node" hostname --ip-address)

echo "Head node: \$head_node"
echo "Head node IP: \$head_node_ip"

export head_node_ip="\$head_node_ip"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME="\${HF_HOME:-\$SCRATCH/huggingface}"
# Load HF token from HF_HOME/token if present and export for non-interactive login
if [[ -f "\${HF_HOME%/}/token" ]]; then
    export HF_TOKEN="\$(cat "\${HF_HOME%/}/token")"
    export HUGGINGFACE_HUB_TOKEN="\${HF_TOKEN}"
    echo "Using HF token from \${HF_HOME%/}/token"
else
    echo "No token file at \${HF_HOME%/}/token; ensure HF_TOKEN or HUGGINGFACE_HUB_TOKEN is set"
fi

# Handle IPv6 addresses - convert to IPv4 if needed
if [[ "\$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"\$head_node_ip"
    if [[ \${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=\${ADDR[1]}
    else
        head_node_ip=\${ADDR[0]}
    fi
    echo "IPV6 address detected. Using IPv4 address: \$head_node_ip"
fi

# Start head node
port=6382
head_address=\$head_node_ip:\$port
export RAY_ADDRESS="\$head_address"
export VLLM_HOST_IP="\$head_node_ip"

echo "Head Address (RAY_ADDRESS): \$head_address"
echo "Head VLLM_HOST_IP: \$VLLM_HOST_IP"

echo "Starting HEAD at \$head_node at IP \$head_node_ip"
ray start --head \\
          --node-ip-address=\$head_node_ip \\
          --port=\$port \\
          --num-cpus=\${SLURM_CPUS_PER_TASK} \\
          --num-gpus=4 \\
          --resources="{\"node:\$head_node_ip\": 1}" \\
          --block &
sleep 10

# Start worker nodes
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

echo "========================================"
echo "Ray cluster status:"
ray status
echo "========================================"
unset SSL_CERT_FILE

# Run the annotation script
python -u -m activeuf.oracle.get_raw_annotations_binarized \
    --dataset_path allenai/ultrafeedback_binarized_cleaned \
    --dataset_split ${SPLIT} \
    --model_name="Qwen/Qwen3-235B-A22B" \
    --max_tokens 24000 \
    --output_path ${OUTPUT_PATH} \
    --model_class vllm_server \
    --temperature 0.0 \
    --top_p 0.1 \
    --num_nodes \$num_nodes_per_instance \
    --batch_size ${BATCH_SIZE} ${REASONING_FLAGS} ${DIRECT_OUTPUT_FLAGS} ${BATCH_INTERVAL_FLAGS}

echo "========================================"
echo "Annotation job completed!"
echo "========================================"
EOF

echo ""
echo "Job submitted! Check logs in ./logs/annotation/binarized/"