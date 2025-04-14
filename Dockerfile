FROM nvcr.io/nvidia/pytorch:25.01-py3
       
# Install the rest of dependencies.
RUN pip install vllm==0.8.3
RUN pip install rewardbench
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
    accelerate==1.6.0