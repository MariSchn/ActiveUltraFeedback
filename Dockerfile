FROM nvcr.io/nvidia/pytorch:25.01-py3
WORKDIR /workspace

# Install venv for potential future use (e.g. LM harness)
RUN apt-get update && apt-get install python3-pip python3-venv -y

# Limit the number of jobs to avoid OOM errors.
ENV MAX_JOBS=24
ENV CMAKE_BUILD_PARALLEL_LEVEL=24
ENV MAKEFLAGS="-j24"

# Install vllm from source to avoid overwriting the torch version from the base image (as described in docs)
# ! This step takes very long
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /workspace/vllm
RUN python use_existing_torch.py
RUN pip install -r requirements/build.txt
RUN pip install -v -e . --no-build-isolation

# Install rewardbench separately to avoid version conflicts (exploiting pips dependency resolution)
RUN pip install rewardbench==0.1.3

# Install the rest of the dependencies
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
dotenv==0.9.9