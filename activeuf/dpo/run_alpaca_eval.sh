#!/usr/bin/env bash
# Run AlpacaEval 2 using a named model config.

# Usage:
#   1) git clone https://github.com/tatsu-lab/alpaca_eval.git resources/alpaca_eval
#   2) pip install -e resources/alpaca_eval
#   3) create your config at:
#        resources/alpaca_eval/src/alpaca_eval/models_configs/<config_name>/configs.yaml
#      (e.g., "olmo-1b" with model_name: allenai/olmo-1b)
#   4) chmod +x activeuf/dpo/run_alpaca_eval.sh
#   5) activeuf/dpo/run_alpaca_eval.sh allenai/olmo-1b oasst_pythia_12b
#
# Optional env vars:
#   ANNOTATORS_CONFIG (default: alpaca_eval_gpt4_turbo_fn)
#   OPENAI_API_KEY (required for the default annotator)

set -euo pipefail

# --- paths ---
REPO_DIR="resources/alpaca_eval"
VENV_DIR="$HOME/alpacaeval-venv"

# --- clone if missing ---
if [ ! -d "$REPO_DIR" ]; then
  echo "Cloning AlpacaEval into $REPO_DIR..."
  git clone https://github.com/tatsu-lab/alpaca_eval.git "$REPO_DIR"
else
  echo "Directory $REPO_DIR already exists, skipping git clone."
fi

# --- create/activate venv ---
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --- ensure packaging tooling + compatible deps ---
# - setuptools<81 keeps pkg_resources available (alpaca_eval imports it)
# - datasets 2.x is needed because 3.x+ (and beyond) removed loading scripts support
pip install --upgrade pip
pip install "setuptools<81" wheel
pip install "datasets==2.19.1"
pip install "peft"
pip install "torch"
pip install "transformers"
pip install "protobuf<5" 
pip install "sentencepiece"
pip install "blobfile"

# --- install/update alpaca-eval ---
pushd "$REPO_DIR" >/dev/null
pip install --upgrade pip
pip install -e .
popd >/dev/null

# --- args ---
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hf_model_name_or_config_hint>"
  echo "Example: $0 allenai/olmo-1b"
  exit 1
fi

HF_OR_HINT="$1"
CONFIG_NAME="${HF_OR_HINT##*/}"
CFG_PATH="${REPO_DIR}/src/alpaca_eval/models_configs/${CONFIG_NAME}/configs.yaml"

if [[ ! -f "$CFG_PATH" ]]; then
  echo "ERROR: Expected config at: $CFG_PATH"
  exit 1
fi

ANNOTATORS_CONFIG="${ANNOTATORS_CONFIG:-alpaca_eval_gpt4_turbo_fn}"
OUTDIR="outputs/alpaca_eval/${CONFIG_NAME}"
mkdir -p "$OUTDIR"

echo "Using venv python: $(which python)"
echo "Running AlpacaEval with config: $CONFIG_NAME"
alpaca_eval evaluate_from_model \
  --model_configs "$CONFIG_NAME" \
  --annotators_config "$ANNOTATORS_CONFIG" \
  --save_dir "$OUTDIR"

echo "Done. Results saved under: $OUTDIR"

