#!/bin/bash
# filepath: /iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/resources/olmes/reproducibility-scripts/tulu3_dev/do_everything.sh

# ==============================================================================
# CONFIGURATION
# ==============================================================================
THIS_DIR="${SCRATCH}/ActiveUltraFeedback"
GENERATOR="$THIS_DIR/resources/olmes/reproducibility-scripts/generate_run_script.py"

# Initialize variables (No default paths)
# We use ${VAR:-} to allow environment variables to pass through, 
# but we do NOT hardcode paths here.
DPO_TRAINED_DIR="${DPO_TRAINED_DIR:-}"
CONFIG_DIR="${CONFIG_DIR:-}"
SCRIPT_OUTPUT_DIR="${SCRIPT_OUTPUT_DIR:-}"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-}"

# Optional controls with safe defaults
SUBSAMPLE_COUNT="${SUBSAMPLE_COUNT:-0}" # 0 means "All models"
DRY_RUN="${DRY_RUN:-false}"
LIMIT="${LIMIT:-0}" 

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
help_function() {
    echo "Usage: $0 [options]"
    echo "Required Options:"
    echo "  --dpo_dir <path>       Directory containing DPO trained models"
    echo "  --output_dir <path>    Directory to save generated shell scripts"
    echo "  --results_dir <path>   Base directory for evaluation results"
    echo "  --config_dir <path>    Directory to store temporary JSON configs"
    echo ""
    echo "Optional Options:"
    echo "  --subsample <int>      Number of models to process (default: all)"
    echo "  --dry_run              Generate configs but do not submit jobs"
    echo "  -h, --help             Show this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dpo_dir)
            DPO_TRAINED_DIR="$2"
            shift 2
            ;;
        --output_dir)
            SCRIPT_OUTPUT_DIR="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_BASE_DIR="$2"
            shift 2
            ;;
        --config_dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --subsample)
            SUBSAMPLE_COUNT="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN=true
            shift
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

if [ -z "$DPO_TRAINED_DIR" ]; then
    echo "Error: --dpo_dir is required."
    missing_args=true
fi
if [ -z "$SCRIPT_OUTPUT_DIR" ]; then
    echo "Error: --output_dir is required."
    missing_args=true
fi
if [ -z "$RESULTS_BASE_DIR" ]; then
    echo "Error: --results_dir is required."
    missing_args=true
fi
if [ -z "$CONFIG_DIR" ]; then
    echo "Error: --config_dir is required."
    missing_args=true
fi

if [ "$missing_args" = true ]; then
    echo "----------------------------------------"
    help_function
fi

# Ensure output directories exist
mkdir -p "$CONFIG_DIR"
mkdir -p "$SCRIPT_OUTPUT_DIR"

echo "========================================"
echo "CONFIGURATION"
echo "========================================"
echo "DPO_TRAINED_DIR: $DPO_TRAINED_DIR"
echo "CONFIG_DIR:      $CONFIG_DIR"
echo "SCRIPT_OUTPUT:   $SCRIPT_OUTPUT_DIR"
echo "RESULTS_BASE:    $RESULTS_BASE_DIR"
echo "GENERATOR:       $GENERATOR"
echo "SUBSAMPLE:       ${SUBSAMPLE_COUNT} (0 = All)"
echo "DRY_RUN:         $DRY_RUN"
echo "----------------------------------------"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# All possible tasks
ALL_TASKS=("gsm8k::tulu" "minerva_math::tulu" "ifeval::tulu" "truthfulqa::tulu")

# Checks if the directory contains actual model weights (.safetensors, .bin, .pt)
# It deliberately ignores subdirectories (like checkpoint-*) by using -maxdepth 1
is_valid_model_dir() {
    local dir="$1"
    # Check for existence of any weight file type in the immediate directory
    local has_weights=$(find "$dir" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \) -print -quit)

    if [ -n "$has_weights" ]; then
        return 0 # Valid
    else
        return 1 # Invalid
    fi
}

# Function to check if a task has been processed (folder exists)
task_is_processed() {
    local model_name="$1"
    local task="$2"
    local task_dir_name="${task//::/_}"
    local result_dir="$RESULTS_BASE_DIR/$task_dir_name/$model_name"
    
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
    
    if [ ${#missing_tasks[@]} -gt 0 ]; then
        printf '%s\n' "${missing_tasks[@]}"
    fi
}

# ==============================================================================
# MAIN LOGIC
# ==============================================================================

# 1. Collect all model directories
FINAL_MODELS=()
if [ -d "$DPO_TRAINED_DIR" ]; then
    for model_dir in "$DPO_TRAINED_DIR"/*; do
        FINAL_MODELS+=("$model_dir")
    done
else
    echo "Error: DPO directory '$DPO_TRAINED_DIR' does not exist."
    exit 1
fi

# 2. Apply Subsampling
SUBSAMPLE_MODELS=()
TOTAL_FOUND=${#FINAL_MODELS[@]}

# Determine the loop limit
if [ "$SUBSAMPLE_COUNT" -gt 0 ]; then
    LIMIT_COUNT=$SUBSAMPLE_COUNT
else
    LIMIT_COUNT=$TOTAL_FOUND
fi

# Fill the subsample array
for ((i=0; i<LIMIT_COUNT && i<TOTAL_FOUND; i++)); do
    SUBSAMPLE_MODELS+=("${FINAL_MODELS[$i]}")
done

printf 'Found %d models. Processing %d models.\n' "$TOTAL_FOUND" "${#SUBSAMPLE_MODELS[@]}"
echo "----------------------------------------"

count=0
skipped_count=0
shopt -s nullglob

for model_dir in "${SUBSAMPLE_MODELS[@]}"; do
  [ -d "$model_dir" ] || continue

  # 1. Basic Config Check
  if [ ! -f "$model_dir/config.json" ]; then
    echo "Skipping $model_dir (no config.json found)"
    ((skipped_count++))
    continue
  fi

  # 2. Robust Model Weights Check
  if ! is_valid_model_dir "$model_dir"; then
    echo "Skipping $model_dir (No .safetensors/.bin/.pt found in root)"
    ((skipped_count++))
    continue
  fi

  run_name="$(basename "$model_dir")"
  
  # Check which tasks are missing
  echo ""
  echo "Checking evaluation status for: $run_name"
  
  MISSING_TASKS=()
  while IFS= read -r task; do
    [[ -n "$task" ]] && MISSING_TASKS+=("$task")
  done < <(get_missing_tasks "$run_name")
  
  echo "DEBUG: Number of missing tasks = ${#MISSING_TASKS[@]}"
  
  # CRITICAL: Skip if no missing tasks
  if [ ${#MISSING_TASKS[@]} -eq 0 ]; then
    echo "✓✓✓ SKIPPING $run_name - ALL TASKS ALREADY PROCESSED ✓✓✓"
    ((skipped_count++))
    continue
  fi
  
  echo "  >>> Will evaluate these tasks: ${MISSING_TASKS[*]}"

  # Optional runtime LIMIT (separate from subsample)
  if [[ "$LIMIT" -gt 0 && "$count" -ge "$LIMIT" ]]; then
    echo "Reached processing LIMIT=$LIMIT. Stopping."
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
  
  # Write JSON config
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

  # Try to find the generated script path
  script_path="$(echo "$gen_output" | grep -Eo '(/[^[:space:]]+\.sh)\b' | tail -n1)"

  # Fallback search
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
  chmod +x "$script_path"
  
  # Execute the script
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
echo "Total models in directory: $TOTAL_FOUND"
echo "Subsample size: ${#SUBSAMPLE_MODELS[@]}"
echo "Models processed (new evals submitted): $count"
echo "Models skipped (already complete or invalid): $skipped_count"