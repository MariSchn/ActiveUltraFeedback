from dotenv import load_dotenv
import huggingface_hub
import json
import os
import time
from typing import Generator, Union, List, Optional, Tuple

import numpy as np
import random
import torch
import openai

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from transformers import pipeline, Pipeline, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

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
    
def load_model(
        model_name: str, 
        max_num_gpus: Optional[int] = None, 
        model_class: Optional[str] = "transformers",
        model_kwargs: Optional[dict] = {}
    ) -> Union[
        Tuple[None, None],                          # model requires API calls (e.g. gpt-4)
        Tuple[AutoModelForCausalLM, AutoTokenizer], # model_class == "transformers"
        Tuple[Pipeline, None],                      # model_class == "pipeline"
        Tuple[LLM, AnyTokenizer],                   # model_class == "vllm"
    ]:
    """
    Loads a model given the name.
    The `model_name` must be one of the keys in the `MODEL_MAP` dictionary (configs.py) or a model_name that uses API calls (e.g. gpt-4).
    In case the model requires API calls and is not available locally (e.g. gpt-4), the function will simply return None, as there is no model to be loaded.

    Args:
        model_name (str): The name of the model to load, which is a key in the `MODEL_MAP` dictionary (configs.py).
        max_num_gpus (Optional[int]): The maximum number of GPUs to use for loading the model (only used for vLLM models).
        model_class (Optional[str]): The class of the model to load. This determines the type of the output. Must be one of ["transformers", "pipeline", "vllm"].
        model_kwargs (Optional[dict]): Additional keyword arguments to pass to the model when loading it. 
    Returns:
        Union[Tuple[None, None], Tuple[AutoModelForCausalLM, AutoTokenizer], Tuple[Pipeline, None], Tuple[LLM, AnyTokenizer]]: The loaded model and tokenizer (if applicable).
    """
    if model_name in ["gpt-3", "gpt-4"]:
        return None, None

    # get HF model path
    model_path = MODEL_MAP[model_name]

    # determine model params
    tensor_parallel_size = torch.cuda.device_count()
    if isinstance(max_num_gpus, int):
        tensor_parallel_size = min(max_num_gpus, tensor_parallel_size)

    dtype = MODEL2DTYPE.get(model_name, "auto")

    # load model
    model = None
    tokenizer = None
        
    if model_class == "transformers":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto",
            # TODO: Ideally set this to "auto" but it causes some models to throw errors during loading or inference, so just leave empty (float32) for now
            # torch_dtype="auto", 
            # * Avoid sliding window attention warning (this warning only occurs for Qwen2.5 models. But the code on the model card also does not do this)
            # attn_implementation="flash_attention_2",  
            **model_kwargs
        )
        # padding_side should be "left" for text generation (https://huggingface.co/docs/transformers/llm_tutorial)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left"
        )
    elif model_class == "pipeline":
        model = pipeline(
            "text-generation", 
            model=model_path, 
            torch_dtype=dtype,
            device_map="auto",
            **model_kwargs
        )
    elif model_class == "vllm":
        model = LLM(
            model_path, 
            gpu_memory_utilization=0.95, 
            swap_space=1, 
            tensor_parallel_size=tensor_parallel_size, 
            trust_remote_code=True, 
            dtype=dtype,
            **model_kwargs
        )
        tokenizer = model.get_tokenizer()
    else:
        raise ValueError(f"Invalid model_class: {model_class}. Must be one of ['transformers', 'pipeline', 'vllm']")
    
    # Check tokenizer and set padding token if needed
    if tokenizer is not None:
        if tokenizer.chat_template is None:
            raise ValueError("Tokenizer does not have a chat template. Please use a model that supports chat templates.")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
            if isinstance(model, PreTrainedModel):
                model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def get_completion(
        prompt: Union[str, List[str]], 
        model: Optional[Union[LLM, Pipeline, PreTrainedModel]] = None, 
        tokenizer: Optional[AutoTokenizer] = None,
        model_name: Optional[str] = None, 
        sampling_params: Optional[SamplingParams] = None, 
        system_prompt: Optional[Union[str, List[str]]] = None, 
        max_api_retry: Optional[int] = MAX_API_RETRY,
        generate_kwargs: Optional[dict] = {}
    ) -> Union[str, List[str]]:   
    """
    This function generates completions for a given prompt using either a local model or API calls (e.g., OpenAI).
    It returns the generated completion/response as a string.

    Batched generation is supported, i.e. passing a list of prompts (and system prompts) instead of a single one and will return a list of completions.
    In case of API calls, the function will retry the API call up to `max_api_retry` times in case of an error.

    Args:
        prompt (Union[str, List[str]]): The prompt(s) to generate completions for.
        model (Optional[Union[LLM, Pipeline, PreTrainedModel]]): The local model to use for generation. If this is not passed the `model_name` needs to be set to a valid model name for API calls.
        tokenizer (Optional[AutoTokenizer]): The tokenizer to use for generation. This is only needed if the model is a "transformers" model (AutoModelForCausalLM).
        model_name (Optional[str]): The name of the model to use for generation (e.g., "gpt-4"). Note that this has to match a model used for an API call or the `model` parameters needs to be set.
        sampling_params (Optional[SamplingParams]): The sampling parameters for generation.
        system_prompt (Optional[Union[str, List[str]]]): The system prompt(s) to use for generation.
        max_api_retry (Optional[int]): The maximum number of retries for API calls.
        generate_kwargs (Optional[dict]): Additional keyword arguments to pass to generate function of the model.
    Returns:
        Union[str, List[str]]: The generated completion(s) for the prompt(s).
    """
    # Check parameters
    if model_name is None and model is None:
        raise ValueError("Either model_name or model must be provided.")
    elif isinstance(model, PreTrainedModel) and tokenizer is None:
        raise ValueError("Tokenizer must be provided if model is an AutoModelForCausalLM (PreTrainedModel).")

    # Make sure `prompt` is a list even in the case of a single prompt
    if isinstance(prompt, str):
        prompt = [prompt]

    # Make sure `system_prompt` is a list of the same length as `prompt` if it is provided
    if system_prompt is not None:
        if isinstance(system_prompt, str):
            system_prompt = [system_prompt] * len(prompt)
        if len(system_prompt) != len(prompt):
            raise ValueError("Length of system_prompt and prompt must match.")
    
        chats = [
            [
                {"role": "system", "content": system_prompt[i]},
                {"role": "user", "content": prompt[i]}
            ]
            for i in range(len(prompt))
        ]
            
    # Generate completions
    if isinstance(model, PreTrainedModel):
        # Tokenize
        if system_prompt is not None:
            prompt = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True) 

        # Pad to multiple of 8 to avoid potential misalignment (https://github.com/google-deepmind/gemma/issues/169)  
        inputs = tokenizer(prompt, padding=True, pad_to_multiple_of=8, return_tensors="pt",).to(model.device)
        padded_input_length = inputs.input_ids.shape[1]

        # Generate
        outputs = model.generate(
            **inputs,
            do_sample=True,  # Required for temperature and top_p to work
            temperature=sampling_params.temperature, 
            top_p=sampling_params.top_p, 
            max_new_tokens=sampling_params.max_tokens,
            **generate_kwargs
        )

        # AutoModelForCausalLM does not allow to return only the generated text so manually remove the input
        outputs = outputs[:, padded_input_length:]
        content = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    elif isinstance(model, Pipeline):
        content = model(
            chats if system_prompt is not None else prompt, 
            return_full_text=False,
            num_return_sequences=1, 
            temperature=sampling_params.temperature, 
            top_p=sampling_params.top_p, 
            max_new_tokens=sampling_params.max_tokens, 
            **generate_kwargs
            )
        content = [response[0]["generated_text"] for response in content]
    elif isinstance(model, LLM):
        if system_prompt is not None:
            # Generate completions using the `chat` method
            responses = model.chat(chats, sampling_params=sampling_params, **generate_kwargs)
            content = [response.outputs[0].text for response in responses]
        else:
            # No `system_prompt` -> Generate completions using the `generate` method
            responses = model.generate(prompt, sampling_params=sampling_params, **generate_kwargs)
            content = [response.outputs[0].text for response in responses]
    elif "gpt" in model_name:
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
                    frequency_penalty=0,
                    **generate_kwargs
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