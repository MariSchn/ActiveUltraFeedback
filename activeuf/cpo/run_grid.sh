#!/bin/bash

# USE_LORA=true
# LORA_PAIRS=(
#     "64 128"
#     "16 32"
#     "64 16"
# )

LEARNING_RATES=("1e-07" "1e-06" "1e-05") 

BETAS=("1.0" "1.5" "2.0")
GAMMAS=("0.5" "1.0" "1.4")

PER_DEVICE_BATCH=4
GRAD_ACCUM_STEPS=4

SBATCH_SCRIPT="activeuf/cpo/training.sbatch"
BASE_MODEL_DIR="$SCRATCH/models/cpo"

echo "Staging Grid Search..."

process_run() {
    local rank=$1
    local alpha=$2
    local suffix=$3 
    
    local RunName="-lr${lr}-sg${gamma}-b${beta}${suffix}"
    
    if ls -d "$BASE_MODEL_DIR"/*"$RunName" 1> /dev/null 2>&1; then
        echo "SKIPPING: Found existing run matching *${RunName}"
        return
    fi
        
    sbatch \
        --export=ALL,MyLR="$lr",MyBeta="$beta",MyGamma="$gamma",MyBS="$PER_DEVICE_BATCH",MyGAS="$GRAD_ACCUM_STEPS",MyUseLora="$USE_LORA",MyRank="$rank",MyAlpha="$alpha" \
        $SBATCH_SCRIPT
        
    echo "Submitted run: $RunName"
    sleep 0.3
}

for lr in "${LEARNING_RATES[@]}"; do
  for beta in "${BETAS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do

        if [ "$USE_LORA" = true ]; then
            for pair in "${LORA_PAIRS[@]}"; do
                set -- $pair
                process_run "$1" "$2" "-loraR$1-loraA$2"
            done
        else
            process_run "" "" "-full"
        fi

    done
  done
done