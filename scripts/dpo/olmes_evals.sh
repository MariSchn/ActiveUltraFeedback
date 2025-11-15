#!/bin/bash
# filepath: /iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/resources/olmes/reproducibility-scripts/tulu3_dev/do_everything.sh

# THIS SCRIPT IS SUPPOSED TO BE RUN FROM THE INSIDE THE OLMES REPOSITORY TO AUTOMATE THE EVALUATION OF MODELS
# MORE DETAILS IN THE loop_experiments.md FILE


# Configurable paths
DPO_TRAINED_DIR="${DPO_TRAINED_DIR:-/iopsstor/scratch/cscs/dmelikidze/models/dpo/active/centered_cosine_big_batches/}"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$THIS_DIR/configs/post-trained/active_experiments_centered"
GENERATOR="$THIS_DIR/../generate_run_script.py"
SCRIPT_OUTPUT_DIR="${SCRATCH:-/iopsstor/scratch/cscs/dmelikidze}/ActiveUltraFeedback/olmes/run/outputs"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/olmes/run/active_new_centered_cosine_big_batches/}"

# Controls
DRY_RUN="${DRY_RUN:-false}"
LIMIT="${LIMIT:-0}"

mkdir -p "$CONFIG_DIR"
mkdir -p "$SCRIPT_OUTPUT_DIR"

echo "DPO_TRAINED_DIR: $DPO_TRAINED_DIR"
echo "CONFIG_DIR:      $CONFIG_DIR"
echo "GENERATOR:       $GENERATOR"
echo "SCRIPT_OUTPUT:   $SCRIPT_OUTPUT_DIR"
echo "RESULTS_BASE:    $RESULTS_BASE_DIR"
echo "DRY_RUN:         $DRY_RUN"
echo "LIMIT:           ${LIMIT:-0}"
echo "----------------------------------------"

# All possible tasks
ALL_TASKS=("gsm8k::tulu" "minerva_math::tulu" "ifeval::tulu" "truthfulqa::tulu")

# Function to check if a task has been processed (folder exists)
task_is_processed() {
    local model_name="$1"
    local task="$2"
    local task_dir_name="${task//::/_}"
    local result_dir="$RESULTS_BASE_DIR/$task_dir_name/$model_name"
    
    # Check if the result directory exists
    if [ -d "$result_dir" ]; then
        return 0  # Already processed
    fi
    
    return 1  # Not processed
}

# Function to get list of missing tasks for a model
get_missing_tasks() {
    local model_name="$1"
    local missing_tasks=()
    
    for task in "${ALL_TASKS[@]}"; do
        if task_is_processed "$model_name" "$task"; then
            echo "  ✓ Already processed: $task" >&2
        else
            echo "  ✗ Missing evaluation for task: $task" >&2
            missing_tasks+=("$task")
        fi
    done
    
    # Return missing tasks as array (via stdout)
    # Only print if there are missing tasks
    if [ ${#missing_tasks[@]} -gt 0 ]; then
        printf '%s\n' "${missing_tasks[@]}"
    fi
}

FINAL_MODELS=()
for model_dir in "$DPO_TRAINED_DIR"/*; do
  FINAL_MODELS+=("$model_dir")
done

SUBSAMPLE_MODELS=()
# TAKING elements [1:5)
for ((i=0; i<100 && i<${#FINAL_MODELS[@]}; i++)); do
    SUBSAMPLE_MODELS+=("${FINAL_MODELS[$i]}")
done

printf 'Subsampled models to evaluate (%d):\n' "${#SUBSAMPLE_MODELS[@]}"
printf '  %s\n' "${SUBSAMPLE_MODELS[@]}"
echo "----------------------------------------"

count=0
skipped_count=0
shopt -s nullglob

for model_dir in "${SUBSAMPLE_MODELS[@]}"; do
  [ -d "$model_dir" ] || continue

  # Skip if config.json is not present (indicates incomplete training)
  if [ ! -f "$model_dir/config.json" ]; then
    echo "Skipping $model_dir (no config.json found)"
    ((skipped_count++))
    continue
  fi

  # Skip dirs that contain only a single .json file (and nothing else)
  json_count=$(find "$model_dir" -mindepth 1 -maxdepth 1 -type f -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
  other_count=$(find "$model_dir" -mindepth 1 -maxdepth 1 ! -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$json_count" -eq 1 ] && [ "$other_count" -eq 0 ]; then
    echo "Skipping $model_dir (contains only one .json file)"
    ((skipped_count++))
    continue
  fi

  run_name="$(basename "$model_dir")"
  
  # Check which tasks are missing
  echo ""
  echo "Checking evaluation status for: $run_name"
  
  # Use readarray to properly handle empty output
  MISSING_TASKS=()
  while IFS= read -r task; do
    [[ -n "$task" ]] && MISSING_TASKS+=("$task")
  done < <(get_missing_tasks "$run_name")
  
  echo "DEBUG: Number of missing tasks = ${#MISSING_TASKS[@]}"
  echo "DEBUG: Missing tasks array: '${MISSING_TASKS[*]}'"
  
  # CRITICAL: Skip if no missing tasks
  if [ ${#MISSING_TASKS[@]} -eq 0 ]; then
    echo "✓✓✓ SKIPPING $run_name - ALL TASKS ALREADY PROCESSED ✓✓✓"
    ((skipped_count++))
    continue
  fi
  
  echo "  >>> Will evaluate these tasks: ${MISSING_TASKS[*]}"

  # Optional limit
  if [[ "$LIMIT" -gt 0 && "$count" -ge "$LIMIT" ]]; then
    echo "Reached LIMIT=$LIMIT. Stopping."
    break
  fi

  cfg_path="$CONFIG_DIR/${run_name}.json"
  echo "Preparing config for: $run_name"
  
  # Build JSON array for tasks
  TASKS_JSON="["
  for i in "${!MISSING_TASKS[@]}"; do
    TASKS_JSON+="\"${MISSING_TASKS[$i]}\""
    if [ $i -lt $((${#MISSING_TASKS[@]} - 1)) ]; then
      TASKS_JSON+=","
    fi
  done
  TASKS_JSON+="]"
  
  echo "DEBUG: TASKS_JSON = $TASKS_JSON"
  
  # Write JSON config with only missing tasks
  cat > "$cfg_path" <<EOF
{
    "model_name": "$run_name",
    "checkpoint_path": "$model_dir",
    "output_dir": "$RESULTS_BASE_DIR",
    "script_output_dir": "$SCRIPT_OUTPUT_DIR",
    "wandb_run_path": "ActiveUF/olmes-evals",
    "sbatch_time": "2:30:00",
    "batch_size": 1,
    "eval_script_path": "$SCRATCH/ActiveUltraFeedback/resources/olmes/installation/unattended-eval.sh",
    "tasks": $TASKS_JSON,
    "task_args": {
        "mmlu:mc::tulu": {
            "gpu-memory-utilization": 0.75
        },
        "minerva_math::tulu": {
            "sbatch_time": "8:00:00"
        },
        "bbh:cot-v1::tulu": {
            "sbatch_time": "4:00:00"
        }
    },
    "model_args": {
        "tensor_parallel_size": 4,
        "max_length": 4096,
        "add_bos_token": false
    },
    "extra_olmes_args": [
        "--use-chat-format=True"
    ]
}
EOF

  echo "Wrote config: $cfg_path"

  # Generate run script via the generator
  gen_cmd=(python "$GENERATOR" -c "$cfg_path")
  echo "Running generator: ${gen_cmd[*]}"
  
  if [[ "$DRY_RUN" == "true" ]]; then
    ((count++))
    echo "DRY_RUN: Skipping generator execution and script run."
    echo "----------------------------------------"
    continue
  fi

  gen_output="$("${gen_cmd[@]}" 2>&1 | tee /dev/stderr || true)"

  # Try to find the generated script path from generator output
  script_path="$(echo "$gen_output" | grep -Eo '(/[^[:space:]]+\.sh)\b' | tail -n1)"

  # Fallback: search in SCRIPT_OUTPUT_DIR for the latest script mentioning run_name
  if [[ -z "$script_path" ]]; then
    script_path="$(find "$SCRIPT_OUTPUT_DIR" -maxdepth 2 -type f -name "*${run_name}*.sh" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')"
  fi

  if [[ -z "$script_path" || ! -f "$script_path" ]]; then
    echo "WARNING: Could not locate generated script for $run_name. Skipping execution."
    echo "----------------------------------------"
    ((count++))
    continue
  fi

  echo "Executing generated script: $script_path"
  continue 
  # exit 0
  chmod +x "$script_path"
  
  # Execute the script - don't exit on error
  bash "$script_path"
  script_exit_code=$?
  
  if [ $script_exit_code -ne 0 ]; then
    echo "WARNING: Script exited with code $script_exit_code for $run_name"
  fi

  echo "Submitted/Executed: $run_name"
  echo "----------------------------------------"
  ((count++))
done

echo ""
echo "========================================="
echo "EVALUATION SUMMARY"
echo "========================================="
echo "Total models in subsample: ${#SUBSAMPLE_MODELS[@]}"
echo "Models processed (new evals submitted): $count"
echo "Models skipped (already complete): $skipped_count"
echo "Done. Processed $count model(s), skipped $skipped_count model(s)."