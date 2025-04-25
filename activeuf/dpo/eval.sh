#!/bin/bash

# Check args for pretrained model path
if [ "$#" -ne 3 ]; then
  echo "Three args are required: (1) Pretrained model path, (2) Results dir, (3) LM Eval dir"
  exit 1
fi
PRETRAINED_MODEL_PATH=$1
RESULTS_DIR=$2
LM_EVAL_DIR=$3

# Determine path to LM Eval venv
LM_EVAL_VENV_PATH="$LM_EVAL_DIR/.lm_eval_venv"

# Activate LM Eval venv (if it doesn't exist, create it and install dependencies)
python -m venv $LM_EVAL_VENV_PATH
source $LM_EVAL_VENV_PATH/bin/activate
python -m pip install -e $LM_EVAL_DIR
python -m pip install protobuf tiktoken blobfile langdetect immutabledict

# Run lm_eval
lm_eval --model hf \
    --model_args pretrained=$PRETRAINED_MODEL_PATH,parallelize=True \
    --tasks ifeval \
    --output_path $RESULTS_DIR \
    --batch_size 2 \
    --limit 5 \

# Deactivate venv
deactivate