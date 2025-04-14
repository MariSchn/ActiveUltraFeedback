FROM nvcr.io/nvidia/pytorch:25.01-py3

# Install venv for potential future use (e.g. LM harness)
RUN apt-get update && apt-get install python3-pip python3-venv -y

# Separately install dependencies to avoid conflicts
RUN pip install vllm==0.8.3 --no-deps # ! Can not install dependencies as vllm will overwrite torch with CPU version
RUN pip install rewardbench==0.1.3
RUN pip install \
transformers==4.47.1 \
tokenizers==0.21.1 \
tiktoken==0.9.0 \
datasets==3.4.1 \
openai==1.70.0 \
numpy==1.26.4 \
pandas==2.2.3 \
tqdm==4.67.1 \
deepspeed==0.16.5 \
accelerate==1.6.0 \
msgspec==0.19.0 \
blake3==1.0.4 \
dotenv==0.9.9