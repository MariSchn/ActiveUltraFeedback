from dotenv import load_dotenv
import huggingface_hub
import logging
import os

import numpy as np
import random
import torch

from vllm import LLM, SamplingParams

from activeuf.configs import *
from activeuf.schemas import *

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

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

def sample_principle(source: str) -> str:
    principle_pool = PROMPT_SOURCE2PRINCIPLES.get(source, [DEFAULT_PRINCIPLE])
    principle = random.choice(principle_pool)

    if principle == "honesty":
        if "verbalized_calibration" in PRINCIPLES and np.random.rand() < 0.9:
            principle = "verbalized_calibration"

    return principle

def sample_system_prompt(principle: str) -> str:
    return random.choice(PRINCIPLE2SYSTEM_PROMPTS[principle])
    
def load_model(model_path: str, max_num_gpus: int = None) -> LLM:
    # Determine model params
    tensor_parallel_size = torch.cuda.device_count()
    if isinstance(max_num_gpus, int):
        tensor_parallel_size = min(max_num_gpus, tensor_parallel_size)

    # Instantiate model
    model = LLM(
        model_path, 
        gpu_memory_utilization=0.95, 
        swap_space=1, 
        tensor_parallel_size=tensor_parallel_size, 
        trust_remote_code=True, 
        dtype="auto",
    )

    # Only support models with chat templates
    tokenizer = model.get_tokenizer()
    assert tokenizer.chat_template

    return model

def get_response_texts(
        all_messages: list[list[dict[str, str]]],
        model: str | LLM,
        sampling_params: SamplingParams | None,
        # max_api_retry: int = MAX_API_RETRY,
    ) -> list[str | None]:

    if isinstance(model, str):
        raise ValueError("Response generation with model APIs is not implemented yet.")
    
    elif isinstance(model, LLM):
        all_responses = model.chat(
            all_messages, sampling_params=sampling_params)
        all_response_texts = [_.outputs[0].text for _ in all_responses]

    else:
        raise ValueError("Model must be either a string or an LLM instance.")

    return all_response_texts

if __name__ == "__main__":
    setup(login_to_hf=True)
    set_seed(42)