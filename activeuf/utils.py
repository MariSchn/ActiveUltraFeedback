from dotenv import load_dotenv
import huggingface_hub
import os
import time
from typing import Generator

import numpy as np
import random
import torch
import openai

from vllm import LLM, SamplingParams

from activeuf.configs import *
from activeuf.schemas import *

def setup(login_to_hf: bool = False) -> None:
    # load env variables
    load_dotenv(PUBLIC_ENV_PATH)

    if login_to_hf:
        load_dotenv(LOCAL_ENV_PATH)
        huggingface_hub.login(os.getenv("HF_TOKEN"))

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def yield_samples(filepath: str) -> Generator[Sample, None, None]:
    with open(filepath, "r") as f_in:
        for line in f_in:
            yield Sample(**json.loads(line))

def load_samples(filepath: str) -> list[Sample]:
    samples = []
    with open(filepath, "r") as f_in:
        for line in f_in:
            samples.append(Sample(**json.loads(line)))
    return samples

def sample_principle_for_dataset(dataset_name: str) -> str:
    principle_pool = DATASET2PRINCIPLE_POOL.get(dataset_name, [DEFAULT_PRINCIPLE])
    principle = random.choice(principle_pool)

    if principle == "honesty":
        if "verbalized_calibration" in PRINCIPLES and np.random.rand() < 0.9:
            principle = "verbalized_calibration"

    return principle
    
def load_model(model_name: str, max_num_gpus: int = None) -> LLM:
    # get HF model path
    model_path = MODEL_MAP[model_name]

    # determine model params
    tensor_parallel_size = torch.cuda.device_count()
    if isinstance(max_num_gpus, int):
        tensor_parallel_size = min(max_num_gpus, tensor_parallel_size)

    dtype = MODEL2DTYPE.get(model_name, "auto")
        
    # instantiate model
    model = LLM(
        model_path, 
        gpu_memory_utilization=0.95, 
        swap_space=1, 
        tensor_parallel_size=tensor_parallel_size, 
        trust_remote_code=True, 
        dtype=dtype,
    )

    # if no default chat template exists, load the custom one
    tokenizer = model.get_tokenizer()
    if not tokenizer.chat_template:
        tokenizer.chat_template = MODEL2CHAT_TEMPLATE[model_name]

    return model

def get_response(system_prompt: str, user_prompt: str, model_name: str, sampling_params: SamplingParams, model: LLM = None, max_api_retry: int = MAX_API_RETRY) -> str:    
    if model_name == "gpt-4":
        for _ in range(max_api_retry):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                    temperature=sampling_params.temperature,
                    max_tokens=sampling_params.max_tokens,
                    top_p=sampling_params.top_p,
                    presence_penalty=0,
                    frequency_penalty=0
                )
                content = response.choices[0].message.content
            except Exception as e:
                print(e)
                time.sleep(1)
            else:
                break
    elif model is not None:
        content = model.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            sampling_params=sampling_params,
        )

    return content


if __name__ == "__main__":
    setup(login_to_hf=True)
    set_seed(42)