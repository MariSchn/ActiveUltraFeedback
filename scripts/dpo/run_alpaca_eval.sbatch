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

# Paths
PROJECT_DIR="${SCRATCH}/ActiveUltraFeedback"
ALPACA_EVAL_DIR="${PROJECT_DIR}/resources/alpaca_eval"
HF_CACHE="${HF_CACHE:-${SCRATCH}/huggingface}"

# Path to model to be evaluated and the config for it (this is the same for all models)
MODEL_NAME="${MODEL_NAME:-dpo/1sgwtpjp/v9no9em2}"
MODEL_CONFIG_NAME="tulu-3-sft-8b"

# vLLM server configuration for annotator
ANNOTATOR_PORT="${ANNOTATOR_PORT:-25125}"
ANNOTATOR_API_KEY="${ANNOTATOR_API_KEY:-token-abc123}"
ANNOTATOR_MODEL_NAME="${ANNOTATOR_MODEL_NAME:-meta-llama/Llama-3.3-70B-Instruct}"
ANNOTATOR_GPU_MEM_UTILIZATION="${ANNOTATOR_GPU_MEM_UTILIZATION:-0.7}"
ANNOTATOR_TENSOR_PARALLEL="${ANNOTATOR_TENSOR_PARALLEL:-4}"
ANNOTATOR_CONFIG="${ANNOTATOR_CONFIG:-weighted_alpaca_eval_vllm_llama3_70b}"

# ==============================================================================
# SETUP
# ==============================================================================
set -e  # Exit on error

# Change to project directory
cd "${PROJECT_DIR}"

# Prepare results directory
RESULTS_DIR="${PROJECT_DIR}/results/alpaca_eval/${MODEL_NAME}"
mkdir -p "${RESULTS_DIR}"

# ==============================================================================
# STEP 1: Clone alpaca_eval repo (if needed)
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
# STEP 2: Start vLLM server in background
# ==============================================================================
echo "Starting vLLM server..."
ANNOTATOR_LOG="${RESULTS_DIR}/annotator_server.log"

# Start vLLM server
vllm serve "${ANNOTATOR_MODEL_NAME}" \
    --gpu-memory-utilization "${ANNOTATOR_GPU_MEM_UTILIZATION}" \
    --swap-space 1 \
    --tensor-parallel-size "${ANNOTATOR_TENSOR_PARALLEL}" \
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
# STEP 5: Wait for vLLM server to be ready
# ==============================================================================
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=600      # 10 minutes max wait
WAIT_INTERVAL=10  # Check every 5 seconds
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
    echo "Last 50 lines of vLLM log:"
    tail -n 50 "${ANNOTATOR_LOG}"
    exit 1
fi

# ==============================================================================
# STEP 6: Run evaluation
# ==============================================================================

cd "${PROJECT_DIR}"

# Run the evaluation
alpaca_eval evaluate_from_model \
    --model_configs "${MODEL_CONFIG_NAME}" \
    --annotators_config "${ANNOTATOR_CONFIG}" \
    --output_path "${RESULTS_DIR}"

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
    echo "Results saved to: ${RESULTS_DIR}"
    echo "=========================================="
else
    echo "=========================================="
    echo "Evaluation failed with exit code: ${EVAL_EXIT_CODE}"
    echo "=========================================="
fi

exit ${EVAL_EXIT_CODE}

