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

# 2. Configuration (Adjust path to your python script if needed)
PYTHON_UPDATER_SCRIPT="scripts/update_wandb_run.py"
WANDB_PROJECT="loop"
WANDB_ENTITY="ActiveUF"

# 3. Resolve Paths
# Ensure SCRATCH env var is present
if [ -z "$SCRATCH" ]; then
    echo "Error: \$SCRATCH environment variable is not set."
    exit 1
fi

RM_BASE_DIR="$SCRATCH/models/reward_models/$SWEEP_ID"
DPO_BASE_DIR="$SCRATCH/models/dpo/$SWEEP_ID"

# 4. Verify Directories Exist
if [ ! -d "$RM_BASE_DIR" ]; then
    echo "Error: Reward Model directory does not exist: $RM_BASE_DIR"
    exit 1
fi

if [ ! -d "$DPO_BASE_DIR" ]; then
    echo "Error: DPO directory does not exist: $DPO_BASE_DIR"
    exit 1
fi

echo "=========================================================="
echo "Starting Batch Update for Sweep: $SWEEP_ID"
echo "RM Source:  $RM_BASE_DIR"
echo "DPO Source: $DPO_BASE_DIR"
echo "=========================================================="

# 5. The Loop
# We iterate over RM folders and check if a corresponding DPO folder exists
for rm_run_path in "$RM_BASE_DIR"/*; do
    # Check if it is actually a directory
    if [ -d "$rm_run_path" ]; then
        
        # Extract the Run ID (Folder Name)
        run_id=$(basename "$rm_run_path")
        
        # Construct expected path for DPO
        dpo_run_path="$DPO_BASE_DIR/$run_id"

        # Check if the pair exists
        if [ -d "$dpo_run_path" ]; then
            echo "[MATCH] Found pair for Run ID: $run_id"
            echo "        Processing..."

            # EXECUTE PYTHON SCRIPT
            python "$PYTHON_UPDATER_SCRIPT" \
                --run_id "$run_id" \
                --rm_output_dir "$rm_run_path" \
                --dpo_output_dir "$dpo_run_path" \
                --project "$WANDB_PROJECT" \
                --entity "$WANDB_ENTITY"
            
            echo "        Done."
            echo "----------------------------------------------------------"
        else
            echo "[SKIP]  $run_id - Found in RM but MISSING in DPO path."
        fi
    fi
done

echo "Batch update finished."