from dotenv import load_dotenv
import huggingface_hub
import logging
import os
import time
<<<<<<< HEAD
from typing import Union
=======
from typing import Generator, Union, List, Optional, Tuple
>>>>>>> main

import numpy as np
import random
import torch

import openai
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
<<<<<<< HEAD
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Pipeline, PreTrainedModel
=======
from transformers import pipeline, Pipeline, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
>>>>>>> main

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

def load_model(
        model_name: str, 
        model_class: str = "transformers",
        max_num_gpus: int | None = None, 
        **model_kwargs,
    ) -> Union[
        tuple[str, None],                           # model requires API calls (e.g. gpt-4)
        tuple[AutoModelForCausalLM, AutoTokenizer], # model_class == "transformers"
        tuple[Pipeline, None],                      # model_class == "pipeline"
        tuple[LLM, AnyTokenizer],                   # model_class == "vllm"
    ]:
    """
    Loads a model given the name. 
    
<<<<<<< HEAD
    If the specified model is among the supported APIs, no model is actually loaded and the model name is returned.

    Args:
        model_name (str): The name of the model or API to load.
=======
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
>>>>>>> main
        max_num_gpus (Optional[int]): The maximum number of GPUs to use for loading the model (only used for vLLM models).
        model_class (Optional[str]): The class of the model to load. This determines the type of the output. Must be one of ["transformers", "pipeline", "vllm"].
        model_kwargs (Optional[dict]): Additional keyword arguments to pass to the model when loading it. 
    Returns:
<<<<<<< HEAD
        Union[Tuple[str, None], Tuple[AutoModelForCausalLM, AutoTokenizer], Tuple[Pipeline, None], Tuple[LLM, AnyTokenizer]]: The loaded model and tokenizer (if applicable).
    """
    if model_name in MODEL_APIS:
        return model_name, None
=======
        Union[Tuple[None, None], Tuple[AutoModelForCausalLM, AutoTokenizer], Tuple[Pipeline, None], Tuple[LLM, AnyTokenizer]]: The loaded model and tokenizer (if applicable).
    """
    if model_name in ["gpt-3", "gpt-4"]:
        return None, None

    # get HF model path
    model_path = MODEL_MAP[model_name]
>>>>>>> main

    # determine model params
    tensor_parallel_size = torch.cuda.device_count()
    if isinstance(max_num_gpus, int):
        tensor_parallel_size = min(max_num_gpus, tensor_parallel_size)

<<<<<<< HEAD
    # load model and tokenizer
    if model_class == "transformers":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
=======
    dtype = MODEL2DTYPE.get(model_name, "auto")

    # load model
    model = None
    tokenizer = None
        
    if model_class == "transformers":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
>>>>>>> main
            device_map="auto",
            # TODO: Ideally set this to "auto" but it causes some models to throw errors during loading or inference, so just leave empty (float32) for now
            # torch_dtype="auto", 
            # * Avoid sliding window attention warning (this warning only occurs for Qwen2.5 models. But the code on the model card also does not do this)
            # attn_implementation="flash_attention_2",  
            **model_kwargs
        )
        # padding_side should be "left" for text generation (https://huggingface.co/docs/transformers/llm_tutorial)
        tokenizer = AutoTokenizer.from_pretrained(
<<<<<<< HEAD
            model_name,
            padding_side="left"
        )
    elif model_class == "vllm":
        model = LLM(
            model_name, 
=======
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
>>>>>>> main
            gpu_memory_utilization=0.95, 
            swap_space=1, 
            tensor_parallel_size=tensor_parallel_size, 
            trust_remote_code=True, 
<<<<<<< HEAD
            dtype="auto",
            **model_kwargs
        )
        tokenizer = model.get_tokenizer()
    elif model_class == "pipeline":
        model = pipeline(
            "text-generation", 
            model=model_name, 
            torch_dtype="auto",
            device_map="auto",
            **model_kwargs
        )
        tokenizer = None
=======
            dtype=dtype,
            **model_kwargs
        )
        tokenizer = model.get_tokenizer()
>>>>>>> main
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

<<<<<<< HEAD
def get_response_texts(
        model: str | PreTrainedModel | LLM | Pipeline,
        tokenizer: AutoTokenizer | None,
        all_messages: list[list[dict[str, str]]],
        sampling_params: SamplingParams | None,
        max_api_retry: int = MAX_API_RETRY,
        **generate_kwargs,
    ) -> list[str]:
=======
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
>>>>>>> main
    """
    This function generates responses for the given messages using the specified model.
    The model may be the name of a supported model API (e.g. gpt-4) or a locally loaded model.
    It returns the generated responses.

    Args:
<<<<<<< HEAD
        model (str | PreTrainedModel | LLM | Pipeline): The model to use for generation. This can be a string (e.g. gpt-4), a PreTrainedModel, an LLM, or a Pipeline.
        tokenizer (AutoTokenizer | None): The tokenizer to use for the model. This is required if the model is a PreTrainedModel.
        all_messages (list[list[dict[str, str]]]): The messages to generate responses for. Each message is a list of dictionaries with "role" and "content" keys.
        sampling_params (SamplingParams | None): The sampling parameters to use for generation. This includes temperature, max_tokens, and top_p.
        max_api_retry (int): The maximum number of retries for API calls in case of failure.
        **generate_kwargs: Additional keyword arguments to pass to the model during generation.
=======
        prompt (Union[str, List[str]]): The prompt(s) to generate completions for.
        model (Optional[Union[LLM, Pipeline, PreTrainedModel]]): The local model to use for generation. If this is not passed the `model_name` needs to be set to a valid model name for API calls.
        tokenizer (Optional[AutoTokenizer]): The tokenizer to use for generation. This is only needed if the model is a "transformers" model (AutoModelForCausalLM).
        model_name (Optional[str]): The name of the model to use for generation (e.g., "gpt-4"). Note that this has to match a model used for an API call or the `model` parameters needs to be set.
        sampling_params (Optional[SamplingParams]): The sampling parameters for generation.
        system_prompt (Optional[Union[str, List[str]]]): The system prompt(s) to use for generation.
        max_api_retry (Optional[int]): The maximum number of retries for API calls.
        generate_kwargs (Optional[dict]): Additional keyword arguments to pass to generate function of the model.
>>>>>>> main
    Returns:
        list[str]: The generated response text for each message.
    """
<<<<<<< HEAD

    # generate via API
    if isinstance(model, str):
        if "gpt" in model:
            response_texts = []
            for messages in all_messages:
                for _ in range(max_api_retry):
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=messages,
                            temperature=sampling_params.temperature,
                            max_tokens=sampling_params.max_tokens,
                            top_p=sampling_params.top_p,
                            presence_penalty=0,
                            frequency_penalty=0,
                            **generate_kwargs
                        )
                        response_text = response.choices[0].message.content
                    except Exception as e:
                        print(e)
                        time.sleep(1)
                    else:
                        response_texts.append(response_text)
                        break
        else:
            raise ValueError(f"Model API {model} is not supported. Supported models are: {MODEL_APIS}")
    
    elif isinstance(model, PreTrainedModel):
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided if model is an AutoModelForCausalLM (PreTrainedModel).")

        # ensure padding_side "left" (https://huggingface.co/docs/transformers/llm_tutorial)
        if tokenizer.padding_side != "left":
            raise ValueError("Tokenizer padding side must be 'left' for text generation.")

        all_messages_with_generation_prompt = tokenizer.apply_chat_template(
            all_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        all_inputs = tokenizer(
            all_messages_with_generation_prompt, 
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ).to(model.device)

        all_outputs = model.generate(
            **all_inputs,
            do_sample=True, # required for temperature and top_p to work
            temperature=sampling_params.temperature,
            max_new_tokens=sampling_params.max_tokens,
            top_p=sampling_params.top_p,
            **generate_kwargs,
        )

        # AutoModelForCausalLM does not allow to return only the generated text so manually remove the input
        all_outputs = all_outputs[:, all_inputs.input_ids.shape[1]:]
        response_texts = tokenizer.batch_decode(all_outputs, skip_special_tokens=True)

    elif isinstance(model, LLM):
        all_outputs = model.chat(
            all_messages, sampling_params=sampling_params, **generate_kwargs)
        response_texts = [_.outputs[0].text for _ in all_outputs]

    elif isinstance(model, Pipeline):
        all_outputs = model(
            all_messages,
            return_full_text=False,
            num_return_sequences=1, 
            temperature=sampling_params.temperature, 
            top_p=sampling_params.top_p, 
            max_new_tokens=sampling_params.max_tokens, 
            **generate_kwargs
        )
        response_texts = [_[0]["generated_text"] for _ in all_outputs]

=======
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
>>>>>>> main
    else:
        raise ValueError(f"Was not able to resolve model to be used for generation. model_name: {model_name}, model: {model}")
    
    return response_texts

if __name__ == "__main__":
    setup(login_to_hf=True)
    set_seed(42)