from dotenv import load_dotenv
import huggingface_hub
import logging
import os
import time
from typing import Union
from tqdm import tqdm

import numpy as np
import random
import torch

import openai
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Pipeline, PreTrainedModel

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
        model_class: str = DEFAULT_MODEL_CLASS,
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
    
    If the specified model is among the supported APIs, no model is actually loaded and the model name is returned.

    Args:
        model_name (str): The name of the model or API to load.
        max_num_gpus (Optional[int]): The maximum number of GPUs to use for loading the model (only used for vLLM models).
        model_class (Optional[str]): The class of the model to load. This determines the type of the output. Must be one of ["transformers", "pipeline", "vllm"].
        model_kwargs (Optional[dict]): Additional keyword arguments to pass to the model when loading it. 
    Returns:
        Union[Tuple[str, None], Tuple[AutoModelForCausalLM, AutoTokenizer], Tuple[Pipeline, None], Tuple[LLM, AnyTokenizer]]: The loaded model and tokenizer (if applicable).
    """
    if model_name in MODEL_APIS:
        return model_name, None

    # determine model params
    tensor_parallel_size = torch.cuda.device_count()
    if isinstance(max_num_gpus, int):
        tensor_parallel_size = min(max_num_gpus, tensor_parallel_size)

    # load model and tokenizer
    if model_class == "transformers":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            # TODO: Ideally set this to "auto" but it causes some models to throw errors during loading or inference, so just leave empty (float32) for now
            # torch_dtype="auto", 
            # * Avoid sliding window attention warning (this warning only occurs for Qwen2.5 models. But the code on the model card also does not do this)
            # attn_implementation="flash_attention_2",  
            **model_kwargs
        )
        # padding_side should be "left" for text generation (https://huggingface.co/docs/transformers/llm_tutorial)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left"
        )
    elif model_class == "vllm":      
        # Search over tensor_parallel_size, as number of attention heads needs to be divisible by it
        tps = tensor_parallel_size
        model = None

        while model is None and tps > 0:
            try:
                model = LLM(
                    model_name, 
                    gpu_memory_utilization=0.95, 
                    swap_space=1, 
                    tensor_parallel_size=tps, 
                    trust_remote_code=True, 
                    dtype="auto",
                    **model_kwargs
                )
            except Exception as e:
                print(f"Failed to load model with tensor_parallel_size={tps}: {e}")
                print(f"Retrying with tensor_parallel_size={tps-1}...")
                tps -= 1
        if model is None:
            raise ValueError(f"Failed to load model {model_name} with any tensor_parallel_size.")
        
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

def get_response_texts(
        model: str | PreTrainedModel | LLM | Pipeline,
        tokenizer: AutoTokenizer | None,
        all_messages: list[list[dict[str, str]]],
        sampling_params: SamplingParams | None,
        batch_size: int = 64,
        max_api_retry: int = MAX_API_RETRY,
        **generate_kwargs,
    ) -> list[str]:
    """
    This function generates responses for the given messages using the specified model.
    The model may be the name of a supported model API (e.g. gpt-4) or a locally loaded model.
    It returns the generated responses.

    Args:
        model (str | PreTrainedModel | LLM | Pipeline): The model to use for generation. This can be a string (e.g. gpt-4), a PreTrainedModel, an LLM, or a Pipeline.
        tokenizer (AutoTokenizer | None): The tokenizer to use for the model. This is required if the model is a PreTrainedModel.
        all_messages (list[list[dict[str, str]]]): The messages to generate responses for. Each message is a list of dictionaries with "role" and "content" keys.
        sampling_params (SamplingParams | None): The sampling parameters to use for generation. This includes temperature, max_tokens, and top_p.
        batch_size (int): The batch size to use for generation. This is only used if the model is a locally loaded model.
        max_api_retry (int): The maximum number of retries for API calls in case of failure.
        **generate_kwargs: Additional keyword arguments to pass to the model during generation.
    Returns:
        list[str]: The generated response text for each message.
    """

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

        batches = [all_messages[i:i + batch_size] for i in range(0, len(all_messages), batch_size)]
        response_texts = []

        for batch in tqdm(batches, desc="Generating responses", total=len(batches)):
            batch_messages_with_generation_prompt = tokenizer.apply_chat_template(
                batch,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_inputs = tokenizer(
                batch_messages_with_generation_prompt, 
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            ).to(model.device)

            batch_outputs = model.generate(
                **batch_inputs,
                do_sample=True, # required for temperature and top_p to work
                temperature=sampling_params.temperature,
                max_new_tokens=sampling_params.max_tokens,
                top_p=sampling_params.top_p,
                **generate_kwargs,
            )

            # AutoModelForCausalLM does not allow to return only the generated text so manually remove the input
            batch_outputs = batch_outputs[:, batch_inputs.input_ids.shape[1]:]
            batch_texts = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

            response_texts.extend(batch_texts)

    elif isinstance(model, LLM):
        # * vLLM performs batching internally
        all_outputs = model.chat(
            all_messages, 
            sampling_params=sampling_params, 
            # chat_template=tokenizer.chat_template, # * Must be set for Gemma-3-1b-it as otherwise vLLM gets stuck in an infinite requests loop, fetching the same request over and over again
            **generate_kwargs
        )
        response_texts = [_.outputs[0].text for _ in all_outputs]

    elif isinstance(model, Pipeline):
        batches = [all_messages[i:i + batch_size] for i in range(0, len(all_messages), batch_size)]
        response_texts = []

        for batch in tqdm(batches, desc="Generating responses", total=len(batches)):
            batch_outputs = model(
                batch,
                return_full_text=False,
                num_return_sequences=1, 
                temperature=sampling_params.temperature, 
                top_p=sampling_params.top_p, 
                max_new_tokens=sampling_params.max_tokens, 
                **generate_kwargs
            )
            response_texts = [_[0]["generated_text"] for _ in batch_outputs]

            response_texts.extend(response_texts)

    else:
        raise ValueError(f"Was not able to resolve model to be used for generation. model: {model}")
    
    return response_texts

if __name__ == "__main__":
    setup(login_to_hf=True)
    set_seed(42)