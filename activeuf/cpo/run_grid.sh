#!/bin/bash

USE_LORA=true
LORA_PAIRS=(
    "64 16"
)

LOSS_TYPE="simpo" #"simpo" #"ipo"
LEARNING_RATES=("5e-06") # "1e-05" "2e-05" "5e-05")

# SimPO
BETAS=("2.0") # 0.01  
GAMMAS=("1.2")

# IPO
# BETAS=("0.01") # "0.5" "1.0")
# GAMMAS=("0.0") # we don't care for IPO, So just set it to 1 value.

# 5 random seeds
SEEDS=(3848573921 249857201 3985729302 1029384756 2837465910)
# SEEDS=(42)

PER_DEVICE_BATCH=4
GRAD_ACCUM_STEPS=1

SBATCH_SCRIPT="activeuf/cpo/training.sbatch"
BASE_MODEL_DIR="$SCRATCH/models/cpo3"

echo "Staging Grid Search..."

process_run() {
    local rank=$1
    local alpha=$2
    local suffix=$3 
    local lr=$4
    local beta=$5
    local gamma=$6
    local seed=$7
    
    local RunSuffix="-${LOSS_TYPE}-lr${lr}-sg${gamma}-b${beta}-seed${seed}${suffix}"
    
    if ls -d "$BASE_MODEL_DIR"/*"$RunSuffix" 1> /dev/null 2>&1; then
        echo "SKIPPING: Found existing run matching *${RunSuffix}"
        return
    fi
        
    sbatch \
        --export=ALL,MyLR="$lr",MyBeta="$beta",MyGamma="$gamma",MyBS="$PER_DEVICE_BATCH",MyGAS="$GRAD_ACCUM_STEPS",MyUseLora="$USE_LORA",MyRank="$rank",MyAlpha="$alpha",MyLossType="$LOSS_TYPE",MySeed="$seed" \
        $SBATCH_SCRIPT
        
    echo "Submitted run: $RunName"
    sleep 0.3
}

for lr in "${LEARNING_RATES[@]}"; do
  for beta in "${BETAS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
        for seed in "${SEEDS[@]}"; do

            if [ "$USE_LORA" = true ]; then
                for pair in "${LORA_PAIRS[@]}"; do
                    # Split the pair string "64 16" into $1 and $2
                    set -- $pair
                    # PASS ALL VARIABLES HERE:
                    process_run "$1" "$2" "-loraR$1-loraA$2" "$lr" "$beta" "$gamma" "$seed"
                done
            else
                # PASS ALL VARIABLES HERE (Empty strings for Rank/Alpha):
                process_run "" "" "-full" "$lr" "$beta" "$gamma" "$seed"
            fi
            
        done
    done
  done
done