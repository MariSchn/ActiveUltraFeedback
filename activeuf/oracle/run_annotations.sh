MODELS=(
"microsoft/phi-4"
"meta-llama/Llama-3.3-70B-Instruct"
"google/gemma-3-27b-it"
"mistralai/Mistral-Large-Instruct-2411"
"Qwen/Qwen3-14B"
"CohereLabs/c4ai-command-a-03-2025"
"mistralai/Mistral-Small-24B-Instruct-2501"
"Qwen/Qwen3-30B-A3B"
"Qwen/Qwen2.5-72B-Instruct"
"allenai/OLMo-2-0325-32B-Instruct"
"meta-llama/Llama-3.1-8B-Instruct"
"google/gemma-3-12b-it"
"allenai/Llama-3.1-Tulu-3-70B"
"moonshotai/Moonlight-16B-A3B-Instruct"
"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
"Qwen/Qwen3-32B"
"nvidia/Llama-3_3-Nemotron-Super-49B-v1"
)

for MODEL in "${MODELS[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --time=06:00:00
#SBATCH --output=/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/logs/annotations3/run_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/logs/annotations3/run_%j.err
#SBATCH --environment=activeuf_dev
#SBATCH --job-name=annotations

python -m activeuf.oracle.get_raw_annotations \
    --dataset_path /iopsstor/scratch/cscs/dmelikidze/datasets/completions_combined \
    --model_name="Qwen/Qwen3-32B" \
    --max_tokens 24000 \
    --output_path /iopsstor/scratch/cscs/dmelikidze/datasets/ultrafeedback_annotated_combined_new_qwen10/ \
    --model_class vllm \
    --temperature 0.0 \
    --top_p 0.1 \
    --model_to_annotate "$MODEL" \
    --batch_size_to_annotate 5000
EOF
done