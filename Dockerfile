FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /workspace

# Limit the number of jobs to avoid OOM errors while building from source
ENV MAX_JOBS=24
ENV CMAKE_BUILD_PARALLEL_LEVEL=24
ENV MAKEFLAGS="-j24"

# Install venv for potential future use (e.g. LM harness) and update pip to the latest version
RUN apt-get update && apt-get install python3-pip python3-venv -y; \
    pip install --upgrade pip; \
    rm -rf /var/lib/apt/lists/*

# ===== INSTALL VLLM FROM SOURCE =====
# Install vllm from source to avoid overwriting the torch version from the base image.
# This follows the instructions in the vLLM docs: https://docs.vllm.ai/en/v0.8.1/getting_started/installation/gpu.html#build-wheel-from-source
# ! This step takes very long
RUN git clone https://github.com/vllm-project/vllm.git; \
    cd vllm; \
    git checkout b6553be1bc75f046b00046a4ad7576364d03c835; \
    python use_existing_torch.py; \
    pip install -r requirements/build.txt; \
    pip install -v -e . --no-build-isolation

# ===== INSTALL OTHER DEPENDENCIES =====
# Install dependencies that need to be installed individualy to avoid version conflicts
RUN pip install git+https://github.com/allenai/reward-bench.git

# Install dependencies from Jetson AI Lab
RUN pip install \
https://pypi.jetson-ai-lab.dev/sbsa/cu128/+f/4ac/c85cb769ef772/bitsandbytes-0.47.0.dev0-cp312-cp312-linux_aarch64.whl#sha256=4acc85cb769ef772374b654ced2ae1bb0c20e6149eb7446636626045e10b70c0
# flashinfer-python

# Install the rest of the dependencies
RUN pip install \
transformers \
tokenizers \
tiktoken \
datasets \
openai \
numpy \
pandas \
tqdm \
deepspeed \
accelerate \
dotenv \
peft \
nvitop \
mmh3 \
tensordict \
langdetect \
nltk \
immutabledict \
omegaconf 

# Install rewarduq
COPY resources/rewarduq /workspace/rewarduq
RUN pip install -e /workspace/rewarduq

# Set final working directory
WORKDIR /workspace
