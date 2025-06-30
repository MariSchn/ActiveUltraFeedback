FROM nvcr.io/nvidia/pytorch:25.02-py3
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
RUN git checkout aa54a7bf7bae7e1db43693470ebe93e3dcd30f9d
RUN python use_existing_torch.py 
RUN pip install -r requirements/build.txt
RUN pip install -v -e . --no-build-isolation

# Install rewardbench separately to avoid version conflicts (exploiting pips dependency resolution)
RUN pip install git+https://github.com/allenai/reward-bench.git

# Install bitsandbytes from Jetson AI Lab
RUN pip install https://pypi.jetson-ai-lab.dev/sbsa/cu128/+f/4ac/c85cb769ef772/bitsandbytes-0.47.0.dev0-cp312-cp312-linux_aarch64.whl#sha256=4acc85cb769ef772374b654ced2ae1bb0c20e6149eb7446636626045e10b70c0

# Install the rest of the dependencies
RUN pip install \
transformers==4.51.3 \
tokenizers==0.21.1 \
tiktoken==0.9.0 \
datasets==3.4.1 \
openai==1.70.0 \
numpy==1.26.4 \
pandas==2.2.3 \
tqdm==4.67.1 \
deepspeed==0.16.5 \
accelerate==1.6.0 \
dotenv==0.9.9 \
peft \
nvitop \
mmh3 \
tensordict \
langdetect \
nltk \
immutabledict 
