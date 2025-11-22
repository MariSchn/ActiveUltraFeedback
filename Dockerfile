FROM nvcr.io/nvidia/pytorch:25.06-py3 AS base

WORKDIR /workspace

ENV MAX_JOBS=24
ENV CMAKE_BUILD_PARALLEL_LEVEL=24
ENV MAKEFLAGS="-j24"

RUN apt-get update -y ;\
    apt-get install -y git curl wget sudo libibverbs-dev

# Install the vLLM requirements
RUN git clone https://github.com/vllm-project/vllm.git; \
    cd vllm; \
    git checkout b8b302cde434df8c9289a2b465406b47ebab1c2d; \
    python use_existing_torch.py; \
    pip install -r requirements/build.txt; \
    pip install -v -e . --no-build-isolation

# Install the rest of the dependencies
RUN pip install \
tokenizers \
tiktoken \
openai \
numpy \
pandas \
tqdm \
wandb \
deepspeed \
dotenv \
peft \
nvitop \
mmh3 \
tensordict \
langdetect \
nltk \
immutabledict \
omegaconf \
blobfile 

RUN pip install --upgrade --force-reinstall transformers==4.57.1 datasets==4.2.0
RUN pip install --no-deps accelerate==1.11.0
RUN pip install --no-deps trl==0.24.0

# Install rewarduq
COPY resources/rewarduq /workspace/rewarduq
# RUN git -C /workspace/rewarduq checkout b4e7019c8d00cb350d9679cfa645b7a6941405fb
RUN pip install -e /workspace/rewarduq

# Explicitly downgrade numpy to make dependencies from base image happy
RUN pip install --force-reinstall numpy==1.26.4