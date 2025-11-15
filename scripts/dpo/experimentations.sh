#!/bin/bash
set -euo pipefail

# Base path as requested (absolute)
BASE_ROOT="$SCRATCH/datasets/7_preference_datasets"
DPO_MODELS_DIR="$SCRATCH/models/dpo"
BASE_CACHE_DIR="$SCRATCH"

BASE_DATASET_PATHS=(
  "$BASE_ROOT/skywork_with_small"
  "$BASE_ROOT/combined_with_small"
  "$BASE_ROOT/ultrafeedback_with_small"
)

# Function to convert dataset path to DPO model name (replicates Python dataname_handler)
dataname_handler() {
    local dir_name="$1"
    local has_seeds="$2"  # "true" or "false"
    
    IFS='/' read -ra parts <<< "$dir_name"
    local name_we_need=""
    
    if [[ "$has_seeds" == "true" ]]; then
        # Judge model (index -4 from end)
        local judge_index=$((${#parts[@]} - 4))
        local judge="${parts[$judge_index]}"
        
        if [[ "$judge" == *"llama_3.3_70b"* ]]; then
            name_we_need+="llama70B_"
        elif [[ "$judge" == *"qwen_3_235b"* ]]; then
            name_we_need+="qwen235B_"
        elif [[ "$judge" == *"rm"* ]]; then
            name_we_need+="rm8Bsky_"
        else
            echo "ERROR: Unknown judge model in path: $judge" >&2
            return 1
        fi
        
        # Prompt source (index -5 from end)
        local prompt_index=$((${#parts[@]} - 5))
        local prompt="${parts[$prompt_index]}"
        
        if [[ "$prompt" == *"ultrafeedback_with_small"* ]]; then
            name_we_need+="allenai_"
        elif [[ "$prompt" == *"skywork_with_small"* ]]; then
            name_we_need+="skywork_"
        elif [[ "$prompt" == *"combined_with_small"* ]]; then
            name_we_need+="combined_"
        else
            echo "ERROR: Unknown prompt source in path: $prompt" >&2
            return 1
        fi
        
        # Last part + second-to-last part
        name_we_need+="${parts[-1]}_${parts[-2]}"
        
    else
        # Judge model (index -2 from end)
        local judge_index=$((${#parts[@]} - 2))
        local judge="${parts[$judge_index]}"
        
        if [[ "$judge" == *"llama_3.3_70b"* ]]; then
            name_we_need+="llama70B_"
        elif [[ "$judge" == *"qwen_3_235b"* ]]; then
            name_we_need+="qwen235B_"
        elif [[ "$judge" == *"rm"* ]]; then
            name_we_need+="rm8Bsky_"
        else
            echo "ERROR: Unknown judge model in path: $judge" >&2
            return 1
        fi
        
        # Prompt source (index -3 from end)
        local prompt_index=$((${#parts[@]} - 3))
        local prompt="${parts[$prompt_index]}"
        
        if [[ "$prompt" == *"ultrafeedback_with_small"* ]]; then
            name_we_need+="allenai_"
        elif [[ "$prompt" == *"skywork_with_small"* ]]; then
            name_we_need+="skywork_"
        elif [[ "$prompt" == *"combined_with_small"* ]]; then
            name_we_need+="combined_"
        else
            echo "ERROR: Unknown prompt source in path: $prompt" >&2
            return 1
        fi
        
        # Last part only
        name_we_need+="${parts[-1]}"
    fi
    
    echo "$name_we_need"
}

# Function to check if dataset is already trained
is_already_trained() {
    local dataset_path="$1"
    local has_seeds="false"
    
    # Check if path contains "fixed_seed" to determine has_seeds
    if [[ "$dataset_path" == *"fixed_seed"* ]]; then
        has_seeds="true"
    fi
    
    # Get the model name using dataname_handler
    local model_name
    model_name=$(dataname_handler "$dataset_path" "$has_seeds")
    
    if [ $? -ne 0 ]; then
        echo "WARNING: Could not generate model name for $dataset_path" >&2
        return 1
    fi
    
    echo "Checking: $model_name <- $dataset_path" >&2
    
    # Check if any directory exists in DPO models folder matching pattern: *-${model_name}
    # The pattern accounts for SLURM_ID prefix like "953335-model_name"
    local matching_dirs=("$DPO_MODELS_DIR"/*-"$model_name")
    
    # Check if glob matched any actual directories
    if [ -d "${matching_dirs[0]}" ]; then
        echo "  ✓ SKIP: Already trained - ${matching_dirs[0]##*/}" >&2
        return 0  # Already trained
    else
        echo "  ✗ NEW: Not trained yet - $model_name" >&2
        return 1  # Not trained yet
    fi
}

FINAL_DATASETS=()

for BASE_DATASET_PATH in "${BASE_DATASET_PATHS[@]}"; do
  [ -d "$BASE_DATASET_PATH" ] || { echo "Skipping missing base path: $BASE_DATASET_PATH"; continue; }

  for judge_dir in "$BASE_DATASET_PATH"/*; do
    [ -d "$judge_dir" ] || continue
    judge_name="$(basename "$judge_dir")"

    # Ignore the specified judge (handle both possible spellings)
    if [[ "$judge_name" == "qwen_3_32b" || "$judge_name" == "qwwen_3_32b" ]]; then
      continue
    fi

    fixed_dir="$judge_dir/fixed_seed"
    max_seed_dir="$judge_dir/max_seed"
    max_min_dir="$judge_dir/max_min"

    if [[ -d "$fixed_dir" ]]; then
      # For each seed, take random and ultrafeedback
      for seed_sub in "$fixed_dir"/*; do
        [ -d "$seed_sub" ] || continue
        
        if [[ -d "$seed_sub/random" ]]; then
            if ! is_already_trained "$seed_sub/random"; then
                FINAL_DATASETS+=("$seed_sub/random")
            fi
        fi
        
        if [[ -d "$seed_sub/ultrafeedback" ]]; then
            if ! is_already_trained "$seed_sub/ultrafeedback"; then
                FINAL_DATASETS+=("$seed_sub/ultrafeedback")
            fi
        fi
      done
      
      # Also include max_seed and max_min if present (once)
      if [[ -d "$max_seed_dir" ]]; then
          if ! is_already_trained "$max_seed_dir"; then
              FINAL_DATASETS+=("$max_seed_dir")
          fi
      fi
      
      if [[ -d "$max_min_dir" ]]; then
          if ! is_already_trained "$max_min_dir"; then
              FINAL_DATASETS+=("$max_min_dir")
          fi
      fi
    else
      # No fixed_seed: include each subdirectory (typically 3), but avoid adding max_min twice
      for sub in "$judge_dir"/*; do
        [ -d "$sub" ] || continue
        if [[ "$sub" != "$max_min_dir" ]]; then
            if ! is_already_trained "$sub"; then
                FINAL_DATASETS+=("$sub")
            fi
        fi
      done
      
      if [[ -d "$max_min_dir" ]]; then
          if ! is_already_trained "$max_min_dir"; then
              FINAL_DATASETS+=("$max_min_dir")
          fi
      fi
    fi
  done
done

echo ""
echo "=== SUMMARY ==="
echo "Discovered ${#FINAL_DATASETS[@]} untrained dataset paths."
printf '  %s\n' "${FINAL_DATASETS[@]}"


# exit 0
# echo $PWD

# Subsample: take only 1st and 5th elements (0-indexed: 0 and 4)
# Subsample: take first 10 elements (0-indexed: 0 to 9)
# SUBSAMPLE_DATASETS=()
# for ((i=80; i<100 && i<${#FINAL_DATASETS[@]}; i++)); do
#     SUBSAMPLE_DATASETS+=("${FINAL_DATASETS[$i]}")
# done

# echo "Subsampled ${#SUBSAMPLE_DATASETS[@]} datasets for testing:"
# printf '  %s\n' "${SUBSAMPLE_DATASETS[@]}"

# Loop through each dataset and submit job
for DATASET_PATH in "${FINAL_DATASETS[@]}"; do
    [ -d "$DATASET_PATH" ] || { echo "Skipping non-directory: $DATASET_PATH"; continue; }
    
    # Set time based on dataset path
    if [[ "$DATASET_PATH" == *"combined_with_small"* ]]; then
        TIME_LIMIT="04:00:00"
    else
        TIME_LIMIT="02:30:00"
    fi
    
    # Set seed based on whether fixed_seed is in path
    if [[ "$DATASET_PATH" == *"fixed_seed"* ]]; then
        SEED=102
    else
        SEED=2837465910
    fi
    
    echo "Submitting job for: $DATASET_PATH (time: $TIME_LIMIT, seed: $SEED)"
    
    sbatch << EOF
#!/bin/bash

#SBATCH -A a-infra01-1
#SBATCH --job-name=dpo_experiments
#SBATCH --output=logs/dpo/O-%x.%j
#SBATCH --error=logs/dpo/E-%x.%j
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=$TIME_LIMIT

export HF_HOME=$BASE_CACHE_DIR/hf_home
export VLLM_CACHE_DIR=$BASE_CACHE_DIR/vllm_cache
export WANDB_PROJECT=DPO
export ACCELERATE_DIR="\${ACCELERATE_DIR:-/accelerate}"

MAIN_PROCESS_IP=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
MAIN_PROCESS_PORT=29500
NUM_PROCESSES=\$(expr \$SLURM_NNODES \\* \$SLURM_GPUS_ON_NODE)

CMD="accelerate launch \\
    --config_file=\$SCRATCH/ActiveUltraFeedback/configs/accelerate/multi_node.yaml \\
    --num_processes \$NUM_PROCESSES \\
    --num_machines \$SLURM_NNODES \\
    --machine_rank \\\$SLURM_NODEID \\
    --main_process_ip \$MAIN_PROCESS_IP \\
    --main_process_port \$MAIN_PROCESS_PORT \\
    -m activeuf.dpo.training \\
    --config_path \$SCRATCH/ActiveUltraFeedback/configs/dpo_training.yaml \\
    --slurm_job_id \$SLURM_JOB_ID \\
    --dataset_path $DATASET_PATH \\
    --beta 0.1 \\
    --learning_rate 2e-5 \\
    --seed $SEED \\
    --num_epochs 3"

echo \$CMD

START=\$(date +%s)

srun --environment=activeuf_dev bash -c "\$CMD"

END=\$(date +%s)
DURATION=\$(( END - START ))

echo "Job ended at: \$(date)"
echo "Total execution time: \$DURATION seconds"
EOF

    sleep 1
done