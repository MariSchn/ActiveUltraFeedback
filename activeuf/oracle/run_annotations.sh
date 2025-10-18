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
"HuggingFaceTB/SmolLM2-1.7B-Instruct"
"Qwen/Qwen2.5-0.5B-Instruct"
"HuggingFaceTB/SmolLM2-135M-Instruct"
google/gemma-3-1b-it
google/gemma-3-4b-it
Qwen/Qwen3-0.6B
Qwen/Qwen3-1.7B
microsoft/Phi-4-mini-instruct
meta-llama/Llama-3.2-1B-Instruct
meta-llama/Llama-3.2-3B-Instruct
)

for MODEL in "${MODELS[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --time=12:00:00
#SBATCH --output=./logs/annotation/llama_3.3_70b/%j.out
#SBATCH --environment=activeuf_dev
#SBATCH --job-name=annotation

export HF_HOME=/iopsstor/scratch/cscs/smarian/hf_cache

python -m activeuf.oracle.get_raw_annotations \
    --dataset_path /iopsstor/scratch/cscs/smarian/datasets/3_merged_completions/ultrafeedback_with_small \
    --model_name="meta-llama/Llama-3.3-70B-Instruct" \
    --max_tokens 24000 \
    --output_path /iopsstor/scratch/cscs/smarian/datasets/4_annotated_completions/ultrafeedback_with_small/llama_3.3_70b \
    --model_class vllm \
    --temperature 0.0 \
    --top_p 0.1 \
    --model_to_annotate "$MODEL" \
    --batch_size_to_annotate 5000
EOF

    echo "Submitted job for model: $MODEL using Llama 3.3 70B for ultrafeedback"

    sleep 10

    sbatch <<EOF
#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --time=12:00:00
#SBATCH --output=./logs/annotation/qwen_3_32b/%j.out
#SBATCH --environment=activeuf_dev
#SBATCH --job-name=annotation

export HF_HOME=/iopsstor/scratch/cscs/smarian/hf_cache

python -m activeuf.oracle.get_raw_annotations \
    --dataset_path /iopsstor/scratch/cscs/smarian/datasets/3_merged_completions/ultrafeedback_with_small \
    --model_name="Qwen/Qwen3-32B" \
    --max_tokens 24000 \
    --output_path /iopsstor/scratch/cscs/smarian/datasets/4_annotated_completions/ultrafeedback_with_small/qwen_3_32b \
    --model_class vllm \
    --temperature 0.0 \
    --top_p 0.1 \
    --model_to_annotate "$MODEL" \
    --batch_size_to_annotate 5000
EOF

    echo "Submitted job for model: $MODEL using Qwen 3 32B for ultrafeedback"

    sleep 10

    sbatch <<EOF
#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --time=12:00:00
#SBATCH --output=./logs/annotation/llama_3.3_70b/%j.out
#SBATCH --environment=activeuf_dev
#SBATCH --job-name=annotation

export HF_HOME=/iopsstor/scratch/cscs/smarian/hf_cache

python -m activeuf.oracle.get_raw_annotations \
    --dataset_path /iopsstor/scratch/cscs/smarian/datasets/3_merged_completions/skywork_with_small \
    --model_name="meta-llama/Llama-3.3-70B-Instruct" \
    --max_tokens 24000 \
    --output_path /iopsstor/scratch/cscs/smarian/datasets/4_annotated_completions/skywork_with_small/llama_3.3_70b \
    --model_class vllm \
    --temperature 0.0 \
    --top_p 0.1 \
    --model_to_annotate "$MODEL" \
    --batch_size_to_annotate 5000
EOF

    echo "Submitted job for model: $MODEL using Llama 3.3 70B for skywork"

    sleep 10

    sbatch <<EOF
#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --time=12:00:00
#SBATCH --output=./logs/annotation/qwen_3_32b/%j.out
#SBATCH --environment=activeuf_dev
#SBATCH --job-name=annotation

export HF_HOME=/iopsstor/scratch/cscs/smarian/hf_cache

python -m activeuf.oracle.get_raw_annotations \
    --dataset_path /iopsstor/scratch/cscs/smarian/datasets/3_merged_completions/skywork_with_small \
    --model_name="Qwen/Qwen3-32B" \
    --max_tokens 24000 \
    --output_path /iopsstor/scratch/cscs/smarian/datasets/4_annotated_completions/skywork_with_small/qwen_3_32b \
    --model_class vllm \
    --temperature 0.0 \
    --top_p 0.1 \
    --model_to_annotate "$MODEL" \
    --batch_size_to_annotate 5000
EOF

    echo "Submitted job for model: $MODEL using Qwen 3 32B for skywork"

    sleep 10
done