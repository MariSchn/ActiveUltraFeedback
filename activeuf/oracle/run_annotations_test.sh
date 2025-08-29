MODELS=(
"chosen"
"rejected"
)

for MODEL in "${MODELS[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --time=06:00:00
#SBATCH --output=/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/logs/annotations3/test/run_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/logs/annotations3/test/run_%j.err
#SBATCH --environment=activeuf_dev
#SBATCH --job-name=annotations_test

python -m activeuf.oracle.get_raw_annotations_test \
    --dataset_path allenai/ultrafeedback_binarized_cleaned \
    --model_name="Qwen/Qwen3-32B" \
    --max_tokens 24000 \
    --output_path /iopsstor/scratch/cscs/dmelikidze/datasets/combined_annotations_qwen_test10/ \
    --model_class vllm \
    --temperature 0.0 \
    --top_p 0.1 \
    --model_to_annotate "$MODEL" \
    --batch_size_to_annotate 5000
EOF
done