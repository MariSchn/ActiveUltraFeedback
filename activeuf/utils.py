from dotenv import load_dotenv
import huggingface_hub
import logging
import wandb
import os
import time
from typing import Union
from tqdm import tqdm

import numpy as np
import random
import torch

import openai
import vllm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Pipeline, PreTrainedModel

from activeuf.configs import *
from activeuf.schemas import *

import requests
import httpx

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

def setup(login_to_hf: bool = False, login_to_wandb: bool = False) -> None:
    # load env variables
    load_dotenv(PUBLIC_ENV_PATH)

    if login_to_hf:
        load_dotenv(LOCAL_ENV_PATH)
        huggingface_hub.login(os.getenv("HF_TOKEN"))

    if login_to_wandb:
        load_dotenv(LOCAL_ENV_PATH)
        wandb.login(key=os.getenv("WANDB_TOKEN"))
        
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
        num_nodes: int = 1,
        ping_delay: int = PING_DELAY,
        max_ping_retries: int = MAX_PING_RETRIES,
        model_kwargs: dict = {},
    ) -> Union[
        tuple[str, None],                                                # model requires API calls (e.g. gpt-4) or model_class == "vllm_server"
        tuple[AutoModelForCausalLM, AutoTokenizer],                      # model_class == "transformers"
        tuple[Pipeline, None],                                           # model_class == "pipeline"
        tuple[vllm.LLM, vllm.transformers_utils.tokenizer.AnyTokenizer], # model_class == "vllm"
    ]:
    """
    Loads a model given the name. 
    
    If the specified model is among the supported APIs, no model is actually loaded and the model name is returned.

    Args:
        model_name (str): The name of the model or API to load.
        model_class (Optional[str]): The class of the model to load. This determines the type of the output. Must be one of ["transformers", "pipeline", "vllm"].
        max_num_gpus (Optional[int]): The maximum number of GPUs to use for loading the model (only used for vLLM models).
        num_nodes (int): The number of nodes to use for loading the model. This is only used for vLLM models.
        ping_delay (int): Delay between pings to the vLLM server to check if it is already running (only used for model_class == "vllm_server").
        max_ping_retries (int): Number of retries to check if the vLLM server is running (only used for model_class == "vllm_server").
        model_kwargs (Optional[dict]): Additional keyword arguments to pass to the model when loading it. 
    Returns:
        Union[Tuple[str, None], Tuple[AutoModelForCausalLM, AutoTokenizer], Tuple[Pipeline, None], Tuple[LLM, vllm.transformers_utils.tokenizer.AnyTokenizer]]: The loaded model and tokenizer (if applicable).
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
                vllm_kwargs = {
                    "gpu_memory_utilization": 0.9,
                    "swap_space": 1,
                    "tensor_parallel_size": tps,
                    "pipeline_parallel_size": num_nodes,
                    "trust_remote_code": True,
                    "dtype": "auto",
                    "download_dir": "/iopsstor/scratch/cscs/smarian/hf_cache",
                    **model_kwargs
                }

                # Specify tokenizer mode for Mistral models
                if "mistral" in model_name.lower():
                    vllm_kwargs["tokenizer_mode"] = "mistral"

                model = vllm.LLM(
                    model_name, 
                    **vllm_kwargs
                )
            except Exception as e:
                print(f"Failed to load model with tensor_parallel_size={tps}: {e}")
                print(f"Retrying with tensor_parallel_size={tps-1}...")
                tps -= 1
        if model is None:
            raise ValueError(f"Failed to load model {model_name} with any tensor_parallel_size.")
        
        tokenizer = model.get_tokenizer()
    elif model_class == "vllm_server":
        # Start the vLLM server for the model (logic is similar to the vllm class)
        tps = tensor_parallel_size
        model = None

        while model is None and tps > 0:
            try:
                command =  f"vllm serve {model_name}"
                command +=  " --gpu-memory-utilization 0.9"
                command +=  " --swap-space 1"
                command += f" --tensor-parallel-size {tps}"
                command += f" --pipeline-parallel-size {num_nodes}"
                command +=  " --trust-remote-code"
                command +=  " --dtype auto"
                command += f" --download-dir /iopsstor/scratch/cscs/smarian/hf_cache"
                command += f" --port 8000"  # Default port
                command +=  " --tokenizer-mode=mistral" if "mistral" in model_name.lower() else ""
                command += f" > server_tps_{tps}.log 2>&1 &"

                os.system(command)

                server_ready = False
                for attempt in range(max_ping_retries):
                    try:
                        response = requests.get("http://localhost:8000/ping")
                        if response.status_code == 200:
                            server_ready = True
                            break
                    except Exception as e:
                        print(f"Ping attempt {attempt+1} failed: {e}")
                    time.sleep(ping_delay)

                if not server_ready:
                    raise RuntimeError("vLLM server did not start after maximum retries.")
                else:
                    print("vLLM server is ready.")

                model = "http://localhost:8000"  # Return the URL of the vLLM server
                tokenizer = None                 # No tokenizer needed for vLLM server API calls
                
            except Exception as e:
                print(f"Failed to load model with tensor_parallel_size={tps}: {e}")
                print(f"Retrying with tensor_parallel_size={tps-1}...")
                tps -= 1

        if model is None:
            raise ValueError(f"Failed to load model {model_name} with any tensor_parallel_size.")
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
        if not 'mistral' in model_name.lower() and tokenizer.chat_template is None:
            raise ValueError("Tokenizer does not have a chat template. Please use a model that supports chat templates.")
        if not 'mistral' in model_name.lower() and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
            if isinstance(model, PreTrainedModel):
                model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def get_response_texts(
        model: str | PreTrainedModel | vllm.LLM | Pipeline,
        tokenizer: AutoTokenizer | None,
        all_messages: list[list[dict[str, str]]],
        sampling_params: vllm.SamplingParams | None,
        batch_size: int = 64,
        max_api_retry: int = MAX_API_RETRY,
        generate_kwargs: dict = {},
    ) -> list[str]:
    """
    This function generates responses for the given messages using the specified model.
    The model may be the name of a supported model API (e.g. gpt-4) or a locally loaded model.
    It returns the generated responses.

    Args:
        model (str | PreTrainedModel | vllm.LLM | Pipeline): The model to use for generation. This can be a string (e.g. gpt-4), a PreTrainedModel, an LLM, or a Pipeline.
        tokenizer (AutoTokenizer | None): The tokenizer to use for the model. This is required if the model is a PreTrainedModel.
        all_messages (list[list[dict[str, str]]]): The messages to generate responses for. Each message is a list of dictionaries with "role" and "content" keys.
        sampling_params (vllm.SamplingParams | None): The sampling parameters to use for generation. This includes temperature, max_tokens, and top_p.
        batch_size (int): The batch size to use for generation. This is only used if the model is a locally loaded model.
        max_api_retry (int): The maximum number of retries for API calls in case of failure.
        generate_kwargs: Additional keyword arguments to pass to the model during generation.
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
                            model=model,
                            messages=messages,
                            temperature=sampling_params.temperature,
                            max_tokens=sampling_params.max_tokens,
                            top_p=sampling_params.top_p,
                            presence_penalty=sampling_params.presence_penalty,
                            frequency_penalty=sampling_params.frequency_penalty,
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
            # Assuming the model is a vLLM server URL
            response_texts = []
            client = openai.OpenAI(
                api_key="EMPTY",
                base_url=f"{model}/v1",
                http_client=httpx.Client(verify=False)
            )

            models = client.models.list()
            model = models.data[0].id

            # TODO: Refactor to use the Batch API
            for messages in tqdm(all_messages, desc="Generating responses"):
                for _ in range(max_api_retry):
                    try:
                        response = client.chat.completions.create(
                            model=model,
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

    elif isinstance(model, vllm.LLM):
        # * vLLM performs batching internally
        try:
            all_outputs = model.chat(
                all_messages, 
                sampling_params=sampling_params, 
                chat_template=tokenizer.chat_template,
                # use_tqdm=False, # to avoid spamming the console with progress bars
                chat_template_kwargs={"enable_thinking": False},  # disable thinking for now
                **generate_kwargs
            )
        except Exception as e:
            print(f"Failed to generate responses with vLLM: {e}\nRetrying without fixed chat template...")
            all_outputs = model.chat(
                all_messages, 
                sampling_params=sampling_params, 
                # use_tqdm=False, # to avoid spamming the console with progress bars
                chat_template_kwargs={"enable_thinking": False},  # disable thinking for now
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

            batch_texts = [_[0]["generated_text"] for _ in batch_outputs]
            response_texts.extend(batch_texts)

    else:
        raise ValueError(f"Was not able to resolve model to be used for generation. model: {model}")
    
    return response_texts

if __name__ == "__main__":
    setup(login_to_hf=True)
    set_seed(42)