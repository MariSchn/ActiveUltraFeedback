#!/bin/bash

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_DIR="${SCRATCH}/ActiveUltraFeedback"
MODELS_DIR=${MODELS_DIR:-${PROJECT_DIR}/models}

# Initialize variables (allow environment variables to pass through)
MODEL_PATH="${MODEL_PATH:-${MODELS_DIR}/dpo/vgw9073q/pf0a9uk8}"
RESULTS_DIR="${RESULTS_DIR:-${MODEL_PATH}/results/alpaca_eval}"
HF_HOME="${HF_HOME:-${SCRATCH}/cache/hf_cache}"

ALPACA_EVAL_DIR="${ALPACA_EVAL_DIR:-${PROJECT_DIR}/resources/alpaca_eval}"
LOGS_DIR="${LOGS_DIR:-${PROJECT_DIR}/logs/alpaca_eval}"

# vLLM server configuration for annotator
ANNOTATOR_NAME="${ANNOTATOR_NAME:-activeuf}"
ANNOTATOR_PORT="${ANNOTATOR_PORT:-25125}"
ANNOTATOR_API_KEY="${ANNOTATOR_API_KEY:-token-abc123}"
ANNOTATOR_GPU_MEM_UTILIZATION="${ANNOTATOR_GPU_MEM_UTILIZATION:-0.7}"
ANNOTATOR_TENSOR_PARALLEL_SIZE="${ANNOTATOR_TENSOR_PARALLEL_SIZE:-4}"

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
help_function() {
    echo "Usage: $0 [options]"
    echo "Required Options:"
    echo "  --model_path <path>           Model path to evaluate (e.g., 'models/dpo/4nqieq7t/gkzji65z')"
    echo "  --results_dir <path>          Base directory for evaluation results"
    echo ""
    echo "Optional Options:"
    echo "  --annotator_gpu_mem_utilization <float>   GPU memory utilization for annotator (default: 0.7)"
    echo "  --annotator_tensor_parallel_size <int>          Tensor parallel size for annotator (default: 4)"
    echo "  -h, --help                    Show this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_DIR="$2"
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
missing_args=0

if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required."
    missing_args=1
fi
if [ -z "$RESULTS_DIR" ]; then
    echo "Error: --results_dir is required."
    missing_args=1
fi

if [ "$missing_args" -eq 1 ]; then
    echo "----------------------------------------"
    help_function
fi

# ==============================================================================
# SETUP
# ==============================================================================
set -e  # Exit on error

# Initialize exit code (will be updated after evaluation)
EVAL_EXIT_CODE=1

# Prepare results and logs directory
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOGS_DIR}"

echo "========================================"
echo "CONFIGURATION"
echo "========================================"
echo "MODEL_PATH:                     ${MODEL_PATH}"
echo "RESULTS_DIR:                    ${RESULTS_DIR}"
echo "ANNOTATOR_GPU_MEM_UTILIZATION:  ${ANNOTATOR_GPU_MEM_UTILIZATION}"
echo "ANNOTATOR_TENSOR_PARALLEL_SIZE: ${ANNOTATOR_TENSOR_PARALLEL_SIZE}"
echo "HF_HOME:                        ${HF_HOME}"
echo "----------------------------------------"

# Change to project directory
cd "${PROJECT_DIR}"

# Delete old results if they exist
if [ -d "${RESULTS_DIR}" ]; then
    echo "Erasing existing results at ${RESULTS_DIR}"
    rm -rf "${RESULTS_DIR}"
fi
mkdir -p "${RESULTS_DIR}"

# ==============================================================================
# STEP 1: Start vLLM server in background
# ==============================================================================
echo "Starting vLLM server..."
ANNOTATOR_LOG="${RESULTS_DIR}/annotator_server.log"

# Start vLLM server
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --gpu-memory-utilization "${ANNOTATOR_GPU_MEM_UTILIZATION}" \
    --swap-space 1 \
    --tensor-parallel-size "${ANNOTATOR_TENSOR_PARALLEL_SIZE}" \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --dtype bfloat16 \
    --port "${ANNOTATOR_PORT}" \
    --api-key "${ANNOTATOR_API_KEY}" \
    --download-dir "${HF_HOME}" \
    > "${ANNOTATOR_LOG}" 2>&1 &

ANNOTATOR_PID=$!
echo "vLLM server started with PID: ${ANNOTATOR_PID}"
echo "Log file: ${ANNOTATOR_LOG}"

# ==============================================================================
# STEP 2: Prepare clean version of alpaca_eval repo
# ==============================================================================
if [ ! -d "${ALPACA_EVAL_DIR}" ]; then
    echo "Cloning alpaca_eval repo"
    git clone https://github.com/tatsu-lab/alpaca_eval.git "${ALPACA_EVAL_DIR}
fi"

# ==============================================================================
# STEP 3: Create annotator and model configs
# ==============================================================================
echo "Creating annotator config..."

ANNOTATOR_CONFIG_DIR="${ALPACA_EVAL_DIR}/src/alpaca_eval/evaluators_configs/${ANNOTATOR_NAME}"
mkdir -p "${ANNOTATOR_CONFIG_DIR}"
cat > "${ANNOTATOR_CONFIG_DIR}/configs.yaml" <<EOF
${ANNOTATOR_NAME}:
  prompt_template: "alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt"
  fn_completions: "openai_completions"
  completions_kwargs:
    model_name: "meta-llama/Llama-3.3-70B-Instruct"
    max_tokens: 1
    temperature: 1 # temperature should be applied for sampling, so that should make no effect.
    logprobs: true
    top_logprobs: 5
    requires_chatml: true
  fn_completion_parser: "logprob_parser"
  completion_parser_kwargs:
    numerator_token: "m"
    denominator_tokens: ["m", "M"]
    is_binarize: false
  completion_key: "completions_all"
  batch_size: 1
EOF
echo "Annotator config created at: ${ANNOTATOR_CONFIG_DIR}"


echo "Creating config for ${MODEL_PATH}..."

# Extract model name as path relative to MODELS_DIR
# Remove MODELS_DIR prefix and any leading slash
MODEL_NAME="${MODEL_PATH#${MODELS_DIR}/}"
MODEL_NAME="${MODEL_NAME#/}"
MODEL_CONFIG_DIR="${ALPACA_EVAL_DIR}/src/alpaca_eval/models_configs/${MODEL_NAME}"
mkdir -p "${MODEL_CONFIG_DIR}"

# Pointing to a blank prompt template on purpose
cat > "${MODEL_CONFIG_DIR}/prompt.txt" <<EOF
<|user|>
{instruction}
<|assistant|>
EOF
cat > "${MODEL_CONFIG_DIR}/configs.yaml" <<EOF
${MODEL_NAME}:
  prompt_template: "${MODEL_CONFIG_DIR}/prompt.txt"
  fn_completions: "vllm_local_completions"
  completions_kwargs:
    model_name: "${MODEL_PATH}"
    model_kwargs:
      tensor_parallel_size: 2
      gpu_memory_utilization: 0.15
      max_model_len: 4096
    max_new_tokens: 2048
    temperature: 0.0
  pretty_name: ${MODEL_NAME}
EOF
echo "Created custom config at: ${MODEL_CONFIG_DIR}"

# ==============================================================================
# STEP 3: Install alpaca_eval
# ==============================================================================
echo "Installing alpaca_eval..."
cd "${ALPACA_EVAL_DIR}"
python -m pip install . --quiet
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
WAIT_INTERVAL=10  # Check every 10 seconds
ELAPSED=0
SERVER_READY=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if grep -q "Application startup complete" "${ANNOTATOR_LOG}" 2>/dev/null; then
        echo "vLLM server is ready!"
        SERVER_READY=1
        break
    fi
    
    # Also check if the API endpoint is responding
    if curl -s "http://localhost:${ANNOTATOR_PORT}/health" >/dev/null 2>&1; then
        echo "vLLM server is responding!"
        SERVER_READY=1
        break
    fi
    
    echo "Waiting... (${ELAPSED}s / ${MAX_WAIT}s)"
    sleep ${WAIT_INTERVAL}
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ $SERVER_READY -eq 0 ]; then
    echo "ERROR: vLLM server did not become ready within ${MAX_WAIT} seconds"
    echo "Last 50 lines of annotator log:"
    tail -n 50 "${ANNOTATOR_LOG}"
    exit 1
fi
# ==============================================================================
# STEP 6: Run evaluation
# ==============================================================================
echo "Running evaluation..."

# Run the evaluation using the custom config (use the directory name, not full path)
alpaca_eval evaluate_from_model \
    --model_configs "${MODEL_NAME}" \
    --annotators_config "${ANNOTATOR_NAME}" \
    --output_path "${RESULTS_DIR}"

EVAL_EXIT_CODE=$?

# ==============================================================================
# CLEANUP
# ==============================================================================
if [ -n "${ANNOTATOR_PID}" ]; then
    echo "Stopping vLLM server..."
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