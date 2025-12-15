#!/bin/bash
#SBATCH --job-name=alpaca_eval
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --environment=activeuf
#SBATCH --account=a-infra01-1
#SBATCH --partition=normal
#SBATCH --time=00:15:00
#SBATCH --output=./logs/alpaca_eval/alpaca_eval_%j.out
#SBATCH --error=./logs/alpaca_eval/alpaca_eval_%j.err

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_DIR="${SCRATCH}/ActiveUltraFeedback"

# Initialize variables (allow environment variables to pass through)
MODELS_DIR="${MODELS_DIR:-models/dpo}"
MODEL_NAME="${MODEL_NAME:-4nqieq7t/gkzji65z}"
RESULTS_DIR="${RESULTS_DIR:-results/alpaca_eval/dpo}"
HF_CACHE="${HF_CACHE:-${SCRATCH}/huggingface}"

# vLLM server configuration for annotator
ANNOTATOR_PORT="${ANNOTATOR_PORT:-25125}"
ANNOTATOR_API_KEY="${ANNOTATOR_API_KEY:-token-abc123}"
ANNOTATOR_MODEL_NAME="${ANNOTATOR_MODEL_NAME:-meta-llama/Llama-3.3-70B-Instruct}"
ANNOTATOR_GPU_MEM_UTILIZATION="${ANNOTATOR_GPU_MEM_UTILIZATION:-0.7}"
ANNOTATOR_TENSOR_PARALLEL_SIZE="${ANNOTATOR_TENSOR_PARALLEL_SIZE:-4}"
ANNOTATOR_CONFIG_NAME="${ANNOTATOR_CONFIG_NAME:-weighted_alpaca_eval_vllm_llama3_70b}"

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
help_function() {
    echo "Usage: $0 [options]"
    echo "Required Options:"
    echo "  --model_name <path>           Model name/path to evaluate (e.g., '1sgwtpjp/v9no9em2')"
    echo "  --results_dir <path>          Base directory for evaluation results"
    echo ""
    echo "Optional Options:"
    echo "  --annotator_config_name <name>     Annotator config name (default: weighted_alpaca_eval_vllm_llama3_70b)"
    echo "  --annotator_model_name <name>      Annotator model name (default: meta-llama/Llama-3.3-70B-Instruct)"
    echo "  --annotator_gpu_mem_utilization <float>   GPU memory utilization for annotator (default: 0.7)"
    echo "  --annotator_tensor_parallel_size <int>          Tensor parallel size for annotator (default: 4)"
    echo "  -h, --help                    Show this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --annotator_config_name)
            ANNOTATOR_CONFIG_NAME="$2"
            shift 2
            ;;
        --annotator_model_name)
            ANNOTATOR_MODEL_NAME="$2"
            shift 2
            ;;
        --annotator_gpu_mem_utilization)
            ANNOTATOR_GPU_MEM_UTILIZATION="$2"
            shift 2
            ;;
        --annotator_tensor_parallel_size)
            ANNOTATOR_TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            help_function
            ;;
        *)
            echo "Unknown argument: $1"
            help_function
            ;;
    esac
done

# ==============================================================================
# VALIDATION
# ==============================================================================
missing_args=false

if [ -z "$MODEL_NAME" ]; then
    echo "Error: --model_name is required."
    missing_args=true
fi
if [ -z "$RESULTS_DIR" ]; then
    echo "Error: --results_dir is required."
    missing_args=true
fi

if [ "$missing_args" = true ]; then
    echo "----------------------------------------"
    help_function
fi

# ==============================================================================
# SETUP
# ==============================================================================
set -e  # Exit on error

# Paths
ALPACA_EVAL_DIR="${PROJECT_DIR}/resources/alpaca_eval"

# Prepare results directory
THIS_RESULTS_DIR="${RESULTS_DIR}/${MODEL_NAME}"
mkdir -p "${THIS_RESULTS_DIR}"

# Ensure logs directory exists
mkdir -p "${PROJECT_DIR}/logs/alpaca_eval"

echo "========================================"
echo "CONFIGURATION"
echo "========================================"
echo "MODEL_NAME:               ${MODEL_NAME}"
echo "MODELS_DIR:               ${MODELS_DIR}"
echo "THIS_RESULTS_DIR:         ${THIS_RESULTS_DIR}"
echo "ANNOTATOR_CONFIG_NAME:         ${ANNOTATOR_CONFIG_NAME}"
echo "ANNOTATOR_MODEL_NAME:          ${ANNOTATOR_MODEL_NAME}"
echo "ANNOTATOR_GPU_MEM_UTILIZATION: ${ANNOTATOR_GPU_MEM_UTILIZATION}"
echo "ANNOTATOR_TENSOR_PARALLEL_SIZE: ${ANNOTATOR_TENSOR_PARALLEL_SIZE}"
echo "HF_CACHE:                 ${HF_CACHE}"
echo "----------------------------------------"

# Change to project directory
cd "${PROJECT_DIR}"

# ==============================================================================
# STEP 1: Start vLLM server in background
# ==============================================================================
echo "Starting vLLM server..."
ANNOTATOR_LOG="${THIS_RESULTS_DIR}/annotator_server.log"

# Start vLLM server
vllm serve "${ANNOTATOR_MODEL_NAME}" \
    --gpu-memory-utilization "${ANNOTATOR_GPU_MEM_UTILIZATION}" \
    --swap-space 1 \
    --tensor-parallel-size "${ANNOTATOR_TENSOR_PARALLEL_SIZE}" \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --trust-remote-code \
    --dtype bfloat16 \
    --port "${ANNOTATOR_PORT}" \
    --api-key "${ANNOTATOR_API_KEY}" \
    --download-dir "${HF_CACHE}" \
    > "${ANNOTATOR_LOG}" 2>&1 &

ANNOTATOR_PID=$!
echo "vLLM server started with PID: ${ANNOTATOR_PID}"
echo "Log file: ${ANNOTATOR_LOG}"

# ==============================================================================
# STEP 2: Clone alpaca_eval repo (if needed)
# ==============================================================================
if [ ! -d "${ALPACA_EVAL_DIR}" ]; then
    echo "Cloning alpaca_eval repo..."
    mkdir -p "${PROJECT_DIR}/resources"
    cd "${PROJECT_DIR}/resources"
    git clone https://github.com/tatsu-lab/alpaca_eval.git
else
    echo "Alpaca eval repo already exists at ${ALPACA_EVAL_DIR}"
fi

# ==============================================================================
# STEP 3: Install alpaca_eval in edit mode
# ==============================================================================
echo "Installing alpaca_eval in edit mode..."
cd "${ALPACA_EVAL_DIR}"
python -m pip install -e . --quiet
cd "${PROJECT_DIR}"

# ==============================================================================
# STEP 4: Setup environment variables
# ==============================================================================
echo "Setting up environment variables..."
unset SSL_CERT_FILE 2>/dev/null || true
export OPENAI_API_BASE="http://localhost:${ANNOTATOR_PORT}/v1"
export OPENAI_API_KEY="${ANNOTATOR_API_KEY}"

# ==============================================================================
# STEP 5: Create custom model config yaml
# ==============================================================================
echo "Creating custom model config yaml..."

CUSTOM_CONFIG_DIR="${ALPACA_EVAL_DIR}/src/alpaca_eval/models_configs/${MODEL_NAME}"
CUSTOM_CONFIG="${CUSTOM_CONFIG_DIR}/configs.yaml"

# Create the config directory if it doesn't exist
mkdir -p "${CUSTOM_CONFIG_DIR}"

# Create the config file with proper content
cat > "${CUSTOM_CONFIG}" <<EOF
${MODEL_NAME}:
  prompt_template: "Mixtral-8x7B-Instruct-v0.1/togetherai_prompt.txt"
  fn_completions: "vllm_local_completions"
  completions_kwargs:
    model_name: "${MODELS_DIR}/${MODEL_NAME}"
    model_kwargs:
      tensor_parallel_size: 2
      gpu_memory_utilization: 0.15
      max_model_len: 4096
    max_new_tokens: 2048
  pretty_name: "${MODEL_NAME}"
EOF

echo "Created custom config at: ${CUSTOM_CONFIG}"
echo "Model path set to: ${MODEL_PATH}"

# ==============================================================================
# STEP 6: Wait for vLLM server to be ready
# ==============================================================================
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=600      # 10 minutes max wait
WAIT_INTERVAL=10  # Check every 10 seconds
ELAPSED=0
SERVER_READY=false

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if grep -q "Application startup complete" "${ANNOTATOR_LOG}" 2>/dev/null; then
        echo "vLLM server is ready!"
        SERVER_READY=true
        break
    fi
    
    # Also check if the API endpoint is responding
    if curl -s "http://localhost:${ANNOTATOR_PORT}/health" >/dev/null 2>&1; then
        echo "vLLM server is responding!"
        SERVER_READY=true
        break
    fi
    
    echo "Waiting... (${ELAPSED}s / ${MAX_WAIT}s)"
    sleep ${WAIT_INTERVAL}
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ "$SERVER_READY" = false ]; then
    echo "ERROR: vLLM server did not become ready within ${MAX_WAIT} seconds"
    echo "Last 50 lines of annotator log:"
    tail -n 50 "${ANNOTATOR_LOG}"
    exit 1
fi

# ==============================================================================
# STEP 7: Run evaluation
# ==============================================================================
echo "Running evaluation..."

# Run the evaluation using the custom config (use the directory name, not full path)
alpaca_eval evaluate_from_model \
    --model_configs "${MODEL_NAME}" \
    --annotators_config "${ANNOTATOR_CONFIG_NAME}" \
    --output_path "${THIS_RESULTS_DIR}"

EVAL_EXIT_CODE=$?

# ==============================================================================
# CLEANUP
# ==============================================================================
if [ -n "${ANNOTATOR_PID}" ]; then
    echo "Stopping vLLM server (PID: ${ANNOTATOR_PID})..."
    kill ${ANNOTATOR_PID} 2>/dev/null || true
    wait ${ANNOTATOR_PID} 2>/dev/null || true
fi

# ==============================================================================
# EXIT
# ==============================================================================
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "Results saved to: ${THIS_RESULTS_DIR}"
    echo "=========================================="
else
    echo "=========================================="
    echo "Evaluation failed with exit code: ${EVAL_EXIT_CODE}"
    echo "=========================================="
fi

exit ${EVAL_EXIT_CODE}
