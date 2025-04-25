from dotenv import load_dotenv
import huggingface_hub
import json
import logging
import os
import time
from typing import Generator, Union, List, Optional

import numpy as np
import random
import torch
import openai

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

def get_completion(
        prompt: Union[str, List[str]], 
        model: Optional[LLM] = None, 
        model_name: Optional[str] = None, 
        sampling_params: Optional[SamplingParams] = None, 
        system_prompt: Optional[Union[str, List[str]]] = None, 
        max_api_retry: Optional[int] = MAX_API_RETRY
    ) -> Union[str, List[str]]:   
    """
    This function generates completions for a given prompt using either a local model or API calls (e.g., OpenAI).
    It returns the generated completion/response as a string.

    Batched generation is supported, i.e. passing a list of prompts (and system prompts) instead of a single one and will return a list of completions.
    In case of API calls, the function will retry the API call up to `max_api_retry` times in case of an error.

    Args:
        prompt (Union[str, List[str]]): The prompt(s) to generate completions for.
        model (Optional[LLM]): The local model to use for generation. If this is not passed the `model_name` needs to be set to a valid model name for API calls.
        model_name (Optional[str]): The name of the model to use for generation (e.g., "gpt-4"). Note that this has to match a model used for an API call or the `model` parameters needs to be set.
        sampling_params (Optional[SamplingParams]): The sampling parameters for generation.
        system_prompt (Optional[Union[str, List[str]]]): The system prompt(s) to use for generation.
        max_api_retry (Optional[int]): The maximum number of retries for API calls.
    Returns:
        Union[str, List[str]]: The generated completion(s) for the prompt(s).
    """
    if model_name is None and model is None:
        raise ValueError("Either model_name or model must be provided.")

    if model is not None:
        if system_prompt is not None:
            # Make sure `prompt` is a list even in the case of a single prompt
            if isinstance(prompt, str):
                prompt = [prompt]

            # Make sure `system_prompt` is a list of the same length as `prompt`
            if isinstance(system_prompt, str):
                system_prompt = [system_prompt] * len(prompt)
            elif len(system_prompt) != len(prompt):
                raise ValueError("Length of system_prompt and prompt must match.")
            
            # Generate completions using the `chat` method

            messages = [
                [
                    {"role": "system", "content": system_prompt[i]},
                    {"role": "user", "content": prompt[i]}
                ]
                for i in range(len(prompt))
            ]
            responses = model.chat(messages, sampling_params=sampling_params)
            content = [response.outputs[0].text for response in responses]
        else:
            if isinstance(prompt, str):
                prompt = [prompt]

            # No `system_prompt` -> Generate completions using the `generate` method
            responses = model.generate(prompt, sampling_params=sampling_params)
            content = [response.outputs[0].text for response in responses]
    elif model_name == "gpt-4":
        for _ in range(max_api_retry):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
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
    else:
        raise ValueError(f"Was not able to resolve model to be used for generation. model_name: {model_name}, model: {model}")

    if len(content) == 1:
        content = content[0]
    else: 
        return content


if __name__ == "__main__":
    setup(login_to_hf=True)
    set_seed(42)