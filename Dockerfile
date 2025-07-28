# ===================================================
#                       BASE
# ===================================================
FROM nvcr.io/nvidia/pytorch:25.06-py3 AS base

WORKDIR /workspace

ENV VLLM_FLASH_ATTN_VERSION=3
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y ;\
    apt-get install -y git curl wget sudo libibverbs-dev

# Install the vLLM requirements
RUN pip install --no-cache-dir \
    accelerate huggingface_hub hf_transfer modelscope

RUN git clone https://github.com/swiss-ai/vllm.git ;\
    cd vllm ;\
    git remote add upstream https://github.com/vllm-project/vllm.git ;\
    git fetch upstream pull/15777/head:pr-15777 ;\
    git switch pr-15777 ;\
    git switch -c v0.9.0.1-merge v0.9.0.1+swissai ;\
    git merge --no-ff --no-edit pr-15777 ;\
    git submodule update --init --recursive ;\
    python use_existing_torch.py

# Install common and CUDA-specific requirements
RUN pip install --no-cache-dir -r /workspace/vllm/requirements/cuda.txt

# ===================================================
#                      BUILDER
# ===================================================
FROM base AS builder

# Install build dependencies
RUN pip install --no-cache-dir -r /workspace/vllm/requirements/build.txt

ENV TORCH_CUDA_ARCH_LIST='9.0+PTX'
ENV VLLM_FA_CMAKE_GPU_ARCHES='90-real'
ENV MAX_JOBS=64
ENV NVCC_THREADS=8

RUN mkdir wheels
RUN touch /workspace/wheels/buildorder.txt

RUN git clone https://github.com/facebookresearch/xformers.git ;\
    cd xformers ;\
    git checkout v0.0.29.post3 ;\
    git submodule update --init --recursive ;\
    python setup.py bdist_wheel --dist-dir=/workspace/wheels ;\
    echo "xformers" >> /workspace/wheels/buildorder.txt

#TODO:Check ~/.cache/flashinfer/90/cached_ops/sampling/build.ninja csrc do they need to be?
ENV FLASHINFER_ENABLE_AOT=1
RUN git clone https://github.com/flashinfer-ai/flashinfer.git ;\
    cd flashinfer ;\
    git checkout v0.2.2.post1 ;\
    git submodule update --init --recursive ;\
    python setup.py bdist_wheel --dist-dir=/workspace/wheels ;\
    echo "flashinfer" >> /workspace/wheels/buildorder.txt

RUN git clone https://github.com/Dao-AILab/flash-attention.git ;\
    cd flash-attention/hopper ;\
    git submodule update --init --recursive ;\
    MAX_JOBS=16 python setup.py bdist_wheel --dist-dir=/workspace/wheels ;\
    echo "flash_attn" >> /workspace/wheels/buildorder.txt

RUN git clone https://github.com/swiss-ai/transformers.git ;\
    cd transformers ;\
    git checkout v4.52.4+swissai ;\
    git submodule update --init --recursive ;\
    python setup.py bdist_wheel --dist-dir=/workspace/wheels ;\
    echo "transformers" >> /workspace/wheels/buildorder.txt

# Build vLLM wheel
RUN cd /workspace/vllm ;\
    python setup.py bdist_wheel --dist-dir=/workspace/wheels ;\
    echo "vllm" >> /workspace/wheels/buildorder.txt

# ===================================================
#                    FINAL BASE
# ===================================================
FROM base

# Clean up packages and final setup
RUN apt-get clean ;\
    rm -rf /var/lib/apt/lists/*

# Install wheels in the order they were built
RUN --mount=type=bind,from=builder,source=/workspace/wheels,target=/workspace/wheels \
    cat /workspace/wheels/buildorder.txt | while read package ; do \
        pip install --no-deps --no-cache-dir --force-reinstall /workspace/wheels/${package}*.whl ;\
    done


# Install the rest of the dependencies
RUN pip install \
# transformers==4.52.4 \
# tokenizers==0.21.2 \
# tiktoken==0.9.0 \
# datasets==3.6.0 \
# openai==1.93.0 \
# numpy==1.26.4 \
# pandas==2.2.3 \
# tqdm==4.67.1 \
wandb==0.21.0 \
deepspeed==0.17.1 \
# accelerate==1.8.1 \
# dotenv==0.9.9 \
# peft==0.16.0 \
nvitop==1.5.1 \
# mmh3==5.1.0 \
tensordict==0.8.3 \
langdetect==1.0.9 \
nltk==3.9.1 \
immutabledict==4.2.1 \
# omegaconf==2.3.0 \
blobfile==3.0.0

# Install rewarduq
COPY resources/rewarduq /workspace/rewarduq
RUN pip install -e /workspace/rewarduq

# Explicitly downgrade numpy to be compatible with torch and vLLM
RUN pip install --force-reinstall numpy==1.26.4