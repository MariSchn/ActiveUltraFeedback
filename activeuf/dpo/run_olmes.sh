#!/bin/bash

# Usage:
# 1. chmod +x activeuf/dpo/run_olmes.sh
# 2. activeuf/dpo/run_olmes.sh <model_name> <task_name> <output_dir> 
# (e.g. activeuf/dpo/run_olmes.sh olmo-1b arc_challenge::olmes my-eval-dir1)

# Validate arguments
if [ $# -ne 3 ]; then
  echo "Usage: $0 <model_name> <task_name> <output_dir>"
  exit 1
fi

MODEL_NAME=$1
TASK_NAME=$2
OUTPUT_DIR=$3
REPO_DIR="resources/olmes"
VENV_DIR=".venv_olmes"

# Step 1: Clone the repo if not present
if [ ! -d "$REPO_DIR" ]; then
  echo "Cloning olmes into $REPO_DIR..."
  git clone https://github.com/allenai/olmes.git "$REPO_DIR"
else
  echo "Directory $REPO_DIR already exists, skipping git clone."
fi

# Step 2: Set up and activate virtualenv
echo "Creating virtual environment in $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Step 3: Install olmes in editable mode
cd "$REPO_DIR"
pip install --upgrade pip
pip install -e .

# Step 4: Run olmes CLI
echo "Running olmes with model '$MODEL_NAME' on task '$TASK_NAME'..."
olmes --model "$MODEL_NAME" --task "$TASK_NAME"::olmes --output-dir "$OUTPUT_DIR"

echo "✅ Completed evaluation for task: $TASK_NAME, model: $MODEL_NAME, output: $OUTPUT_DIR"
