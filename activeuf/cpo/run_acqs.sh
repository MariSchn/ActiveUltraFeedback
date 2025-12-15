#!/bin/bash

# List of full paths to your datasets
DATASETS=(
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/my/actives/datasets/DeltaQuantile_60829"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/my/actives/datasets/DeltaUCB_60829"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/my/actives/datasets/DoubleTS_60829"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/my/actives/datasets/DRTS_60829"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/my/actives/datasets/InfoGain_60829"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/my/actives/datasets/InfoMax_60829"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/my/actives/datasets/MaxMinLCB_60829"

    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/my/baselines/datasets/delta_qwen_60829"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/my/baselines/datasets/maxmin_60829"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/my/baselines/datasets/random_60829"
    "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/my/baselines/datasets/ultrafeedback_60829"
)

SBATCH_SCRIPT="activeuf/cpo/training.sbatch"

echo "Staging Dataset Sweep..."

USE_LORA=true

for dataset in "${DATASETS[@]}"; do
    
    # Extract dataset name for logging (optional)
    ds_name=$(basename "$dataset")
    
    echo "Submitting job for: $ds_name"
    
    # Submit job, passing the path variable
    sbatch \
        --export=ALL,MyLR="7.0e-6",MyDatasetPath="$dataset",MyBeta="10.0",MyGamma="1.4",MyUseLora="$USE_LORA",MyRank="64",MyAlpha="16" \
        "$SBATCH_SCRIPT"
        
    sleep 1
done

echo "All jobs submitted."