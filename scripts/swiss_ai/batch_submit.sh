#!/bin/bash
dataset_path="/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/PolygloToxicityPrompts_wildchat-all"
#model_name_list=("meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-70B-Instruct" "Qwen/Qwen2.5-72B-Instruct" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B")
model_name_list=("meta-llama/Llama-3.1-70B-Instruct" "Qwen/Qwen2.5-72B-Instruct")

for model_name in "${model_name_list[@]}"; do
  echo "${model_name}"

  # Submit without principle
  sbatch <<EOF
#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --partition=debug
#SBATCH --time=1:30:00
#SBATCH --container-writable
#SBATCH --job-name="${model_name}"
#SBATCH --output=./logs/swiss_ai/${model_name}_%j.out
#SBATCH --exclude=nid006438,nid006439,nid006440,nid006441,nid006442,nid006443,nid006444,nid006445,nid006446,nid006447,nid006448,nid006449,nid006450,nid006451,nid006461,nid006462,nid006868,nid007121

export HF_TOKEN=$(cat ~/.hf-token)

srun --environment=activeuf_dev python swiss_ai.py \
  --dataset_path "${dataset_path}" \
  --dataset_split "train" \
  --model_name "${model_name}" \
  --output_path "${dataset_path}/completions/${model_name#*/}.jsonl"
EOF

#  # Submit with principle
#  sbatch <<EOF
##!/bin/bash
##SBATCH --account=a-infra01-1
##SBATCH --partition=normal
##SBATCH --time=12:00:00
##SBATCH --container-writable
##SBATCH --job-name="${model_name}_w_principles"
##SBATCH --output="./logs/swiss_ai/${model_name}_w_principles_%j.out"
##SBATCH --exclude=nid006438,nid006439,nid006440,nid006441,nid006442,nid006443,nid006444,nid006445,nid006446,nid006447,nid006448,nid006449,nid006450,nid006451,nid006461,nid006462,nid006868
#
#export HF_TOKEN=$(cat ~/.hf-token)
#
#srun --environment=activeuf_dev python swiss_ai.py \
#  --dataset_path "${dataset_path}" \
#  --dataset_split "train" \
#  --model_name "${model_name}" \
#  --output_path "${dataset_path}/completions/${model_name#*/}_w_principles.jsonl" \
#  --w_principles
#EOF
done