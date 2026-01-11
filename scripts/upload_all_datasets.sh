#!/usr/bin/env bash
set -euo pipefail

# Configuration
ORG="ActiveUltraFeedback"
SCRIPT_PATH="scripts/statistics/upload_to_hf.py"
COLLECTION="Preference Datasets"

# Methods inside each prompt dataset directory
METHODS=("DRTS" "DeltaUCB" "DTS" "InfoMax" "MaxMinLCB")

# Define 8 prompt dataset directories (update these paths)
PROMPT_DATASETS=(
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_3/final_datasets/dpo"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/tulu_3/final_datasets/rm"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/ultrafeedback/actives/dpo"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/ultrafeedback/actives/rm"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/skywork/actives/dpo"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/skywork/actives/rm"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/combined/actives/dpo"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/combined/actives/rm"
)

# Corresponding repo name prefixes for each prompt dataset
REPO_PREFIXES=(
    "active_tulu3_dpo"
    "active_tulu3_rm"
    "active_ultrafeedback_dpo"
    "active_ultrafeedback_rm"
    "active_skywork_dpo"
    "active_skywork_rm"
    "active_combined_dpo"
    "active_combined_rm"
)

echo "=========================================="
echo "Uploading Prompt Datasets to ${COLLECTION}"
echo "=========================================="

for i in "${!PROMPT_DATASETS[@]}"; do
    base_path="${PROMPT_DATASETS[$i]}"
    repo_prefix="${REPO_PREFIXES[$i]}"
    
    echo ""
    echo "=========================================="
    echo "Processing: ${repo_prefix}"
    echo "=========================================="
    
    for method in "${METHODS[@]}"; do
        folder_path="${base_path}/${method}"
        repo_name="${repo_prefix}_$(echo "$method" | tr '[:upper:]' '[:lower:]')"
        
        # Skip if folder doesn't exist
        if [[ ! -d "$folder_path" ]]; then
            echo "⚠️  Skipping ${method} - folder not found: ${folder_path}"
            continue
        fi
        
        echo ""
        echo "📦 Uploading ${repo_prefix} - ${method}..."
        echo "   Folder: ${folder_path}"
        echo "   Repo: ${ORG}/${repo_name}"
        
        python3 "$SCRIPT_PATH" \
            --folder_path "$folder_path" \
            --repo_name "$repo_name" \
            --org "$ORG" \
            --collection "$COLLECTION"
        
        echo "✅ Completed: ${repo_name}"
    done
done

echo ""
echo "=========================================="
echo "🎉 All datasets uploaded successfully!"
echo "=========================================="
