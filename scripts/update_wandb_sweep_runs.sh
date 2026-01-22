#!/bin/bash

# =============================================================================
# USAGE: ./update_wandb_sweep_runs.sh <SWEEP_ID>
# Example: ./update_wandb_sweep_runs.sh b27bst06
# =============================================================================

# 1. Input Validation
SWEEP_ID=$1

if [ -z "$SWEEP_ID" ]; then
    echo "Error: You must provide a Sweep ID."
    echo "Usage: $0 <sweep_id>"
    exit 1
fi

# 2. Configuration
PYTHON_UPDATER_SCRIPT="scripts/update_wandb_run.py"
WANDB_PROJECT="loop"
WANDB_ENTITY="ActiveUF"

# 3. Resolve Paths
if [ -z "$SCRATCH" ]; then
    echo "Error: \$SCRATCH environment variable is not set."
    exit 1
fi

RM_BASE_DIR="$SCRATCH/models/reward_models/$SWEEP_ID"
DPO_BASE_DIR="$SCRATCH/models/dpo/$SWEEP_ID"
IPO_BASE_DIR="$SCRATCH/models/cpo_ipo/$SWEEP_ID"
SIMPO_BASE_DIR="$SCRATCH/models/cpo_simpo/$SWEEP_ID"

# 4. Check Existence of Base Directories (Informational only, do not exit)
echo "=========================================================="
echo "Checking directories for Sweep: $SWEEP_ID"
echo "=========================================================="

DIRS_TO_SEARCH=""

check_and_add_dir() {
    local type=$1
    local path=$2
    if [ -d "$path" ]; then
        echo " [✓] Found $type Directory: $path"
        DIRS_TO_SEARCH="$DIRS_TO_SEARCH $path"
    else
        echo " [x] Missing $type Directory: $path (Will skip)"
    fi
}

check_and_add_dir "RM"    "$RM_BASE_DIR"
check_and_add_dir "DPO"   "$DPO_BASE_DIR"
check_and_add_dir "IPO"   "$IPO_BASE_DIR"
check_and_add_dir "SimPO" "$SIMPO_BASE_DIR"

if [ -z "$DIRS_TO_SEARCH" ]; then
    echo "----------------------------------------------------------"
    echo "Error: No directories found for this Sweep ID. Exiting."
    exit 0
fi

echo "=========================================================="
echo "Starting Batch Update..."
echo "=========================================================="

# 5. Gather all unique Run IDs from all existing directories
# We use 'find' to list subdirectories in all valid base folders, 
# then 'basename' to get the IDs, and 'sort | uniq' to deduplicate.
RUN_IDS=$(find $DIRS_TO_SEARCH -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sort | uniq)

if [ -z "$RUN_IDS" ]; then
    echo "No runs found inside the existing directories."
    exit 0
fi

# 6. The Loop
for run_id in $RUN_IDS; do
    
    # Initialize arguments and status tracking
    # If the run_id contains hyphens, use the part after the last hyphen
    SANITIZED_RUN_ID="${run_id##*-}"
    CMD_ARGS="--run_id $SANITIZED_RUN_ID --project $WANDB_PROJECT --entity $WANDB_ENTITY"
    FOUND_COMPONENTS=""
    SHOULD_RUN=false

    # Check RM
    if [ -d "$RM_BASE_DIR/$run_id" ]; then
        CMD_ARGS="$CMD_ARGS --rm_output_dir $RM_BASE_DIR/$run_id"
        FOUND_COMPONENTS="$FOUND_COMPONENTS RM"
        SHOULD_RUN=true
    fi

    # Check DPO
    if [ -d "$DPO_BASE_DIR/$run_id" ]; then
        CMD_ARGS="$CMD_ARGS --dpo_output_dir $DPO_BASE_DIR/$run_id"
        FOUND_COMPONENTS="$FOUND_COMPONENTS DPO"
        SHOULD_RUN=true
    fi

    # Check IPO
    if [ -d "$IPO_BASE_DIR/$run_id" ]; then
        CMD_ARGS="$CMD_ARGS --ipo_output_dir $IPO_BASE_DIR/$run_id"
        FOUND_COMPONENTS="$FOUND_COMPONENTS IPO"
        SHOULD_RUN=true
    fi

    # Check SimPO
    if [ -d "$SIMPO_BASE_DIR/$run_id" ]; then
        CMD_ARGS="$CMD_ARGS --simpo_output_dir $SIMPO_BASE_DIR/$run_id"
        FOUND_COMPONENTS="$FOUND_COMPONENTS SimPO"
        SHOULD_RUN=true
    fi

    # Execute if we found at least one component
    if [ "$SHOULD_RUN" = true ]; then
        echo "[UPDATE] $run_id -> $SANITIZED_RUN_ID | Found:$FOUND_COMPONENTS"
        
        # Run the python script with the constructed arguments
        python "$PYTHON_UPDATER_SCRIPT" $CMD_ARGS
    else
        # This branch theoretically unreachable due to how RUN_IDS are gathered, but safe to keep
        echo "[SKIP]   $run_id | No valid subdirectories found."
    fi

done

echo "----------------------------------------------------------"
echo "Batch update finished."