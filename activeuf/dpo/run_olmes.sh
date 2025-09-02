#!/bin/bash

# Usage:
# 1. chmod +x activeuf/dpo/run_olmes.sh
# 2. activeuf/dpo/run_olmes.sh <model_name> <task_name> <output_dir> 
# (e.g. activeuf/dpo/run_olmes.sh olmo-1b arc_challenge::olmes results/my-eval-dir1)

set -euo pipefail

# --- Args ---
if [ $# -ne 3 ]; then
  echo "Usage: $0 <model_name> <task_name> <output_dir>"
  exit 1
fi
MODEL_NAME=$1
TASK_NAME=$2
OUTPUT_DIR_INPUT=$3

REPO_DIR="resources/olmes"
VENV_DIR=".venv_olmes"

# Resolve OUTPUT_DIR to absolute path
mkdir -p "$OUTPUT_DIR_INPUT"
OUTPUT_DIR="$(cd "$OUTPUT_DIR_INPUT" && pwd)"

# Tame BLAS thread spam
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Prevent user/site contamination & stale CUDA libs
unset PYTHONPATH
export PYTHONNOUSERSITE=1

echo "[debug] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
command -v nvidia-smi >/dev/null 2>&1 || { echo "❌ nvidia-smi not found; GPU runtime missing."; exit 1; }
nvidia-smi -L | grep -q 'GPU ' || { echo "❌ No GPUs visible via nvidia-smi."; exit 1; }
DRV_STR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
DRV_MAJOR=${DRV_STR%%.*}; DRV_MINOR=${DRV_STR#*.}; DRV_MINOR=${DRV_MINOR%%.*}
DRV_NUM=$((DRV_MAJOR*100 + DRV_MINOR))
echo "[info] NVIDIA driver: $DRV_STR"

# Create/activate venv with default python (your 3.12)
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi
# Ensure we actually activate the venv python first in PATH
source "$VENV_DIR/bin/activate"

PY_MM=$(python - <<'PY'
import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")
PY
)
echo "[info] Python in venv: $PY_MM"

# Option B requirements: Python >= 3.12 and driver >= 550
if python - <<'PY'
import sys; sys.exit(0 if (sys.version_info[0],sys.version_info[1]) >= (3,12) else 42)
PY
then
  if [ "$DRV_NUM" -lt 550 ]; then
    echo "❌ Driver $DRV_STR < 550; CUDA 12.4 wheels need R550+ on Python $PY_MM."
    echo "   Use Python 3.11 (cu121) or upgrade the driver."
    exit 1
  fi
else
  echo "❌ Option B expects Python ≥ 3.12; detected $PY_MM. Use Option A if needed."
  exit 1
fi

# Clone OLMES repo if missing
if [ ! -d "$REPO_DIR" ]; then
  echo "Cloning olmes into $REPO_DIR..."
  git clone https://github.com/swiss-ai/olmes.git "$REPO_DIR"
else
  echo "Directory $REPO_DIR already exists, skipping git clone."
fi

# 1) Nuke the bad stuff
pip uninstall -y torch torchvision torchaudio ai2-olmo ai2-olmo-core ai2-olmes vllm || true


# 1.5) Put the venv’s torch/lib FIRST on LD_LIBRARY_PATH
export TORCH_LIB_DIR="$VENV_DIR/bin/../lib/python3.12/site-packages/torch/lib"
export TORCH_LIB_DIR="$(python - <<'PY'
import sys, pathlib
p = pathlib.Path(sys.executable).parent.parent / f"lib/python{sys.version_info[0]}.{sys.version_info[1]}/site-packages/torch/lib"
print(p.resolve())
PY
)"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 2) Install the CUDA stack FIRST (cu124) and only from the cu124 index
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
assert torch.version.cuda == "12.4" and torch.cuda.is_available(), "GPU torch not installed!"
PY

# 3) Install OLMO libs with SAFE pins (avoid ai2-olmo-core>=2.1 which pulls torch>=2.6)
pip install --no-cache-dir "numpy<2" "ai2-olmo-core>=1.8.0,<2.1.0" ai2-olmo

# 4) Install your local OLMES WITHOUT deps so it can't downgrade torch
pip install -e resources/olmes --no-deps

# Make sure loader prefers the Torch-bundled CUDA libs over any module toolkits
TORCH_LIB_DIR="$VENV_DIR/lib/python${PY_MM}/site-packages/torch/lib"
if [ -d "$TORCH_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

# Guard: ensure we're importing Torch from THIS venv (not /usr/local/...)
python - <<PY
import sys, torch, os
print("torch file:", torch.__file__)
assert os.path.realpath(torch.__file__).startswith(os.path.realpath("$VENV_DIR")), \
  f"Importing torch from outside venv! {torch.__file__}"
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
PY

# vLLM on ARM (aarch64) is not available as prebuilt wheels; skip and use HF backend.
ARCH=$(uname -m)
USE_VLLM=1
if [ "$ARCH" != "x86_64" ]; then
  echo "[info] Detected $ARCH; skipping vLLM (no ARM wheels). Using Transformers backend."
  USE_VLLM=0
fi

if [ "$USE_VLLM" -eq 1 ]; then
  # x86_64 path: try generic vllm (no cu suffix). Re-pin Torch afterwards if it bumps versions.
  if ! pip install --upgrade --no-cache-dir "vllm>=0.5.0,<0.6.0"; then
    echo "[warn] vLLM install failed; falling back to Transformers backend."
    USE_VLLM=0
  else
    # Re-pin the torch stack (vLLM sometimes pulls torch 2.6)
    pip install --upgrade --no-cache-dir --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
      "torch==${TORCH_VER}" "torchvision==${TVISION_VER}" "torchaudio==${TAUDIO_VER}"
    pip install "numpy<2" || true
  fi
fi

# Final CUDA sanity (and catch any libcudart mismatch early)
python - <<'PY'
import sys, torch
print("post-install:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
    x = torch.randn(2, device='cuda'); y = x @ x.t(); print("tiny op ok:", y.shape)
sys.exit(0 if torch.cuda.is_available() else 42)
PY
if [[ $? -ne 0 ]]; then
  echo "❌ CUDA still unavailable. Likely library mismatch:"
  echo "   - Ensure no CUDA modules are loaded that precede $TORCH_LIB_DIR on LD_LIBRARY_PATH."
  echo "   - Verify 'torch file' path above points inside $VENV_DIR (not /usr/local/...)."
  exit 1
fi

# Determine GPU count
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a IDS <<< "$CUDA_VISIBLE_DEVICES"; GPUS="${#IDS[@]}"
else
  GPUS="$(nvidia-smi -L | grep -c 'GPU ')"
  [[ "$GPUS" -gt 0 ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS-1)))
fi
[[ "$GPUS" -ge 1 ]] || { echo "❌ No GPUs visible to the process."; exit 1; }
TP="$GPUS"

# Build model backend args (HF by default; vLLM only on x86_64 when installed)
if [ "$USE_VLLM" -eq 1 ]; then
  MODEL_ARGS_JSON=$(python - <<PY
import json
print(json.dumps({"tensor_parallel_size": int("$TP"), "gpu_memory_utilization": 0.9}))
PY
)
  BACKEND_ARGS=( --model-type vllm --model-args "$MODEL_ARGS_JSON" --gpus "$GPUS" )
else
  BACKEND_ARGS=( --gpus "$GPUS" )  # OLMES should default to Transformers backend
fi

# Weights & Biases (optional)
export MODEL_WANDB_NAME="${MODEL_WANDB_NAME:-$MODEL_NAME}"
WANDB_ARGS=( --model-wb-name "$MODEL_WANDB_NAME" )
[[ -n "${WANDB_RUN_PATH:-}" ]] && WANDB_ARGS+=( --wandb-run-path "$WANDB_RUN_PATH" )

echo "Running olmes with model '$MODEL_NAME' on task '$TASK_NAME'..."
env | grep -E 'WANDB|MODEL_WANDB|CUDA_VISIBLE_DEVICES' || true

olmes --model "$MODEL_NAME" --task "$TASK_NAME" \
      --output-dir "$OUTPUT_DIR" \
      "${BACKEND_ARGS[@]}" "${WANDB_ARGS[@]}"

echo "✅ Completed evaluation for task: $TASK_NAME, model: $MODEL_NAME"
echo "📁 Results in: $OUTPUT_DIR"




# 1. OLMES evaluation tasks:
#activeuf/dpo/run_olmes.sh olmo-7b core_9mcqa::olmes <dir>
#activeuf/dpo/run_olmes.sh olmo-7b mmlu::olmes <dir>

# 2. OLMES evaluation tasks:
#activeuf/dpo/run_olmes.sh olmo-7b main_suite::olmo1 <dir>
#activeuf/dpo/run_olmes.sh olmo-7b mmlu::olmo1 <dir>

# 3. TÜLU 3 evaluation tasks:
#"tulu_3_dev": Tasks evaluated during development
#"tulu_3_unseen": Held-out task used during final evaluation

# 4. OLMo 2 evaluation tasks:
#"core_9mcqa::olmes": The core 9 multiple-choice tasks from original OLMES standard
#"mmlu:mc::olmes": The MMLU tasks in multiple-choice format
#"olmo_2_generative::olmes": The 5 generative tasks used in OLMo 2 development
#"olmo_2_heldout::olmes": The 5 held-out tasks used in OLMo 2 final evaluation