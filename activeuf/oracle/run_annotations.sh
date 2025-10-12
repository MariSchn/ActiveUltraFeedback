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
"Qwen/Qwen3-235B-A22B"
"allenai/OLMo-2-0325-32B-Instruct"
"meta-llama/Llama-3.1-8B-Instruct"
"google/gemma-3-12b-it"
"allenai/Llama-3.1-Tulu-3-70B"
"moonshotai/Moonlight-16B-A3B-Instruct"
"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
"Qwen/Qwen3-32B"
"nvidia/Llama-3_3-Nemotron-Super-49B-v1"
"nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"
"deepseek-ai/DeepSeek-V3"
"allenai/Llama-3.1-Tulu-3-405B"
)

for MODEL in "${MODELS[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --time=06:00:00
#SBATCH --output=./logs/annotation/%j.out
#SBATCH --environment=activeuf_dev
#SBATCH --job-name=annotation

export HF_HOME=/iopsstor/scratch/cscs/smarian/hf_cache

python -m activeuf.oracle.get_raw_annotations \
    --dataset_path /iopsstor/scratch/cscs/smarian/datasets/2_merged_completions/ultrafeedback \
    --model_name="meta-llama/Llama-3.3-70B-Instruct" \
    --max_tokens 24000 \
    --output_path /iopsstor/scratch/cscs/smarian/datasets/3_annotated_completions/ultrafeedback_llama_3.3_70b \
    --model_class vllm \
    --temperature 0.0 \
    --top_p 0.1 \
    --model_to_annotate "$MODEL" \
    --batch_size_to_annotate 5000
EOF
done