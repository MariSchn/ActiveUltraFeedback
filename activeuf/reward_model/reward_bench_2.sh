#!/bin/bash
MODEL_PATH=""
BATCH_SIZE="64"
MAX_LENGTH="4096"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model <model_path> [--batch_size <size>] [--max_length <length>]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: Please provide the model path."
    echo "Usage: $0 --model <model_path> [--batch_size <size>] [--max_length <length>]"
    exit 1
fi

pip install rewardbench

accelerate launch --config_file=$SCRATCH/ActiveUltraFeedback/configs/accelerate/single_node.yaml ./activeuf/reward_model/reward_bench_2.py --model=$MODEL_PATH --max_length=$MAX_LENGTH --batch_size=$BATCH_SIZE