#!/bin/bash

# Usage:
# 1. chmod +x activeuf/dpo/run_olmes.sh
# 2. activeuf/dpo/run_olmes.sh <model_name> <task_name> <output_dir> 
# (e.g. activeuf/dpo/run_olmes.sh olmo-1b arc_challenge::olmes results/my-eval-dir1)

set -euo pipefail
# Validate arguments
if [ $# -ne 3 ]; then
  echo "Usage: $0 <model_name> <task_name> <output_dir>"
  exit 1
fi

MODEL_NAME=$1
TASK_NAME=$2
OUTPUT_DIR_INPUT=$3
REPO_DIR="resources/olmes"
VENV_DIR=".venv_olmes"

# Resolve OUTPUT_DIR to an absolute path BEFORE we cd anywhere
mkdir -p "$OUTPUT_DIR_INPUT"
OUTPUT_DIR="$(cd "$OUTPUT_DIR_INPUT" && pwd)"

# Optional: tame BLAS thread spam (from your last issue)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Clone if missing
if [ ! -d "$REPO_DIR" ]; then
  echo "Cloning olmes into $REPO_DIR..."
  git clone https://github.com/allenai/olmes.git "$REPO_DIR"
else
  echo "Directory $REPO_DIR already exists, skipping git clone."
fi

# Create/activate venv (only if missing)
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install olmes
pushd "$REPO_DIR" >/dev/null
pip install --upgrade pip
pip install -e .
popd >/dev/null

# Run olmes from anywhere; output goes to the absolute path
echo "Running olmes with model '$MODEL_NAME' on task '$TASK_NAME'..."
olmes --model "$MODEL_NAME" --task "$TASK_NAME" --output-dir "$OUTPUT_DIR"

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