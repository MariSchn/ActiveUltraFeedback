from dataclasses import asdict, is_dataclass, fields
from datetime import datetime
from dotenv import load_dotenv
import huggingface_hub
import inspect
import logging
import wandb
import os
import asyncio
import requests
import httpx
import time
from typing import Any, Union
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
import yaml

import numpy as np
import random
import torch

import openai
import vllm
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    PreTrainedModel,
)

from activeuf.configs import *
from activeuf.schemas import *


def ensure_dataclass(cls, d):
    if not is_dataclass(cls):
        return d  # primitive type, leave as is
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        value = d[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = ensure_dataclass(f.type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def convert_dataclass_instance_to_yaml_str(instance) -> str:
    return yaml.dump(asdict(instance))


def get_timestamp(more_detailed=False) -> str:
    now = datetime.now()
    if more_detailed:
        return now.strftime("%Y%m%d-%H%M%S") + f"-{now.microsecond // 1000:03d}"
    return now.strftime("%Y%m%d-%H%M%S")


def get_logger(name, logs_path="app.log", accelerator=None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False  # prevent double logging

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # Console handler for all ranks
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler only for main process
    if accelerator is None or accelerator.is_main_process:
        file_handler = logging.FileHandler(logs_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup(login_to_hf: bool = False, login_to_wandb: bool = False) -> None:
    # load env variables
    load_dotenv(PUBLIC_ENV_PATH)
    load_dotenv(LOCAL_ENV_PATH)

    if login_to_hf:
        huggingface_hub.login(os.getenv("HF_TOKEN"))

    if login_to_wandb:
        wandb.login(key=os.getenv("WANDB_TOKEN"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # False
    torch.backends.cudnn.benchmark = True  # maybe TRUE
    os.environ["PYTHONHASHSEED"] = str(seed)


def filter_dict(dict_to_filter, func):
    sig = inspect.signature(func)
    valid_keys = {
        param.name
        for param in sig.parameters.values()
        if param.kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return {key: dict_to_filter[key] for key in valid_keys if key in dict_to_filter}


def sample_principle(source: str) -> str:
    principle_pool = PROMPT_SOURCE2PRINCIPLES.get(source, DEFAULT_PRINCIPLES)
    principle = random.choice(principle_pool)

    # if principle == "honesty":
    #     if "verbalized_calibration" in PRINCIPLES and np.random.rand() < 0.9:
    #         principle = "verbalized_calibration"

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
    # model requires API calls (e.g. gpt-4) or model_class == "vllm_server"
    tuple[str, None],
    # model_class == "transformers"
    tuple[AutoModelForCausalLM, AutoTokenizer],
    # model_class == "pipeline"
    tuple[Pipeline, None],
    # model_class == "vllm"
    tuple[vllm.LLM, vllm.transformers_utils.tokenizer.AnyTokenizer],
]:
    """
        Loads a model given the name.

        If the specified model is among the supported APIs, no model is actually loaded and the model name is returned.

        Args:
            model_name (str): The name of the model or API to load.
            model_class (Optional[str]): The class of the model to load. This determines the type of the output. Must be one of ["transformers", "pipeline", "vllm", "vllm_server"].
            max_num_gpus (Optional[int]): The maximum number of GPUs to use for loading the model (only used for vLLM models).
            num_nodes (int): The number of nodes to use for loading the model. This is only used for vLLM models.
            data_parallel_size (Optional[int]): The size of the data parallel group (only applicable for vllm_server model class).
            ping_delay (int): Delay between pings to the vLLM server to check if it is already running (only used for model_class == "vllm_server").
            max_ping_retries (int): Number of retries to check if the vLLM server is running (only used for model_class == "vllm_server").
    <<<<<<< HEAD
            gpu_memory_utilization (float): The GPU memory utilization to use for loading the model (only used for vllm models).
            max_model_len (int): The maximum context length of the model. Pass 0 to use the model's default max length.
    =======
    >>>>>>> loop
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
            **model_kwargs,
        )
        # padding_side should be "left" for text generation (https://huggingface.co/docs/transformers/llm_tutorial)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    elif model_class == "vllm":
        # Search over tensor_parallel_size, as number of attention heads needs to be divisible by it
        tps = tensor_parallel_size
        model = None

        while model is None and tps > 0:
            try:
                vllm_kwargs = {
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "swap_space": 1,
                    "tensor_parallel_size": tps,
                    "pipeline_parallel_size": num_nodes,
                    "trust_remote_code": True,
                    "dtype": "auto",
                    "download_dir": os.getenv("HF_CACHE", None),
                    **model_kwargs,
                }

                if max_model_len > 0:
                    vllm_kwargs["max_model_len"] = max_model_len

                # Specify tokenizer mode for Mistral models
                if "mistral" in model_name.lower():
                    vllm_kwargs["tokenizer_mode"] = "mistral"

                model = vllm.LLM(model_name, **vllm_kwargs)
            except Exception as e:
                print(f"Failed to load model with tensor_parallel_size={tps}: {e}")
                print(f"Retrying with tensor_parallel_size={tps - 1}...")
                tps -= 1
        if model is None:
            raise ValueError(
                f"Failed to load model {model_name} with any tensor_parallel_size."
            )

        tokenizer = model.get_tokenizer()
    elif model_class == "vllm_server":
        # Start the vLLM server for the model (logic is similar to the vllm class)
        tps = tensor_parallel_size
        model = None

        # Check if vLLM server is already running
        server_url = "http://localhost:8000"
        try:
            # Check if server is running
            response = requests.get(f"{server_url}/ping")
            if response.status_code == 200:
                # Check which model is loaded
                models_resp = requests.get(f"{server_url}/v1/models")
                if models_resp.status_code == 200:
                    models_data = models_resp.json()
                    loaded_models = [m["id"] for m in models_data.get("data", [])]
                    # Accept both full repo name and short name
                    model_short_name = model_name.split("/")[-1]
                    if model_name in loaded_models or model_short_name in loaded_models:
                        print(
                            f"vLLM server is already running with model: {loaded_models} reusing existing server"
                        )
                    model = server_url
                    tokenizer = None
        except Exception:
            pass

        while model is None and tps > 0:
            try:
                out_file = f"./logs/server/{model_name.split('/')[-1]}_server_tps_{tps}"
                out_file += (
                    f"_{os.getenv('SLURM_JOB_ID', '')}.out"
                    if os.getenv("SLURM_JOB_ID", None)
                    else ".out"
                )

                command = f"vllm serve {model_name}"
                command += f" --gpu-memory-utilization {gpu_memory_utilization}"
                command += " --swap-space 1"
                command += f" --tensor-parallel-size {tps}"
                command += f" --pipeline-parallel-size {num_nodes}"
                command += f" --data-parallel-size {data_parallel_size}"
                command += " --trust-remote-code"
                command += " --dtype auto"
                command += (
                    f" --download-dir {os.getenv('HF_CACHE', None)}"
                    if os.getenv("HF_CACHE", None)
                    else ""
                )
                command += f" --port 8000"  # Default port
                command += (
                    " --tokenizer-mode=mistral"
                    if "mistral" in model_name.lower()
                    else ""
                )
                command += f" > {out_file} 2>&1"
                command += " &"  # Run in background

                print(f"Starting vLLM server with command: {command}")
                os.system(command)

                server_ready = False
                for attempt in range(max_ping_retries):
                    try:
                        response = requests.get("http://localhost:8000/ping")
                        if response.status_code == 200:
                            server_ready = True
                            break
                    except Exception as e:
                        print(f"Ping attempt {attempt + 1} failed: {e}")
                    time.sleep(ping_delay)

                if not server_ready:
                    raise RuntimeError(
                        "vLLM server did not start after maximum retries."
                    )
                else:
                    print("vLLM server is ready.")

                model = "http://localhost:8000"  # Return the URL of the vLLM server
                tokenizer = None  # No tokenizer needed for vLLM server API calls

            except Exception as e:
                print(f"Failed to load model with tensor_parallel_size={tps}: {e}")
                print(f"Retrying with tensor_parallel_size={tps - 1}...")
                tps -= 1

        if model is None:
            raise ValueError(
                f"Failed to load model {model_name} with any tensor_parallel_size."
            )
    elif model_class == "pipeline":
        model = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype="auto",
            device_map="auto",
            **model_kwargs,
        )
        tokenizer = None
    else:
        raise ValueError(
            f"Invalid model_class: {model_class}. Must be one of ['transformers', 'pipeline', 'vllm']"
        )

    # Check tokenizer and set padding token if needed
    if tokenizer is not None:
        if not "mistral" in model_name.lower() and tokenizer.chat_template is None:
            raise ValueError(
                "Tokenizer does not have a chat template. Please use a model that supports chat templates."
            )
        if not "mistral" in model_name.lower() and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

            if isinstance(model, PreTrainedModel):
                model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


async def vllm_server_inference(
    url: str,
    all_messages: list[list[dict[str, str]]],
    sampling_params: vllm.SamplingParams,
    max_api_retry: int,
    generate_kwargs: dict,
) -> list[str]:
    """
    Helper function to perform asynchronous inference using a vLLM server.

    Args:
        url (str): The URL of the vLLM server.
        all_messages (list[list[dict[str, str]]]): The messages to generate responses for
        sampling_params (vllm.SamplingParams): The sampling parameters to use for generation.
        max_api_retry (int): The maximum number of retries for API calls in case of failure.
        generate_kwargs (dict): Additional keyword arguments to pass to the vLLM server during generation

    Returns:
        list[str]: The generated response text for each message.
    """
    client = openai.AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"{url}/v1",
        http_client=httpx.AsyncClient(verify=False, timeout=httpx.Timeout(None)),
    )

    models = await client.models.list()
    model = models.data[0].id

    concurrency_limit = int(os.getenv("VLLM_SERVER_CONCURRENCY_LIMIT", 50))
    semaphore = asyncio.Semaphore(concurrency_limit)

    print(
        f"Using vLLM server at {url} with model {model}. Concurrency limit: {concurrency_limit}"
    )

    # Define helper function that runs the API calls asynchronously
    async def send_request(conversation):
        async with semaphore:
            for _ in range(max_api_retry):
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=conversation,
                        temperature=sampling_params.temperature,
                        max_tokens=sampling_params.max_tokens,
                        top_p=sampling_params.top_p,
                        presence_penalty=sampling_params.presence_penalty,
                        frequency_penalty=sampling_params.frequency_penalty,
                        logprobs=True if sampling_params.logprobs else False,
                        top_logprobs=sampling_params.logprobs,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    return response
                except Exception as e:
                    print(f"An error occurred: {e}. Retrying...")
                    await asyncio.sleep(1)
            raise RuntimeError(f"Failed to get response after {max_api_retry} retries.")

    tasks = [send_request(chat) for chat in all_messages]
    responses = await async_tqdm.gather(*tasks)

    return [response.choices[0].message.content for response in responses], responses


def get_response_texts(
    model: str | PreTrainedModel | vllm.LLM | Pipeline,
    tokenizer: AutoTokenizer | None,
    all_messages: list[list[dict[str, str]]],
    sampling_params: vllm.SamplingParams | None,
    batch_size: int = 64,
    max_api_retry: int = MAX_API_RETRY,
    generate_kwargs: dict = {},
) -> tuple[list[str], list[Any]]:
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
        list[Any]: The raw responses from the model, which may include additional information such as token IDs or attention scores.
    """

    # generate via API
    if isinstance(model, str):
        if "gpt" in model:
            responses = []
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
                            **generate_kwargs,
                        )
                    except Exception as e:
                        print(e)
                        time.sleep(1)
                    else:
                        responses.append(response)
                        break
            response_texts = [
                response.choices[0].message.content for response in responses
            ]
        else:
            response_texts, responses = asyncio.run(
                vllm_server_inference(
                    model, all_messages, sampling_params, max_api_retry, generate_kwargs
                )
            )

    elif isinstance(model, PreTrainedModel):
        if tokenizer is None:
            raise ValueError(
                "Tokenizer must be provided if model is an AutoModelForCausalLM (PreTrainedModel)."
            )

        # ensure padding_side "left" (https://huggingface.co/docs/transformers/llm_tutorial)
        if tokenizer.padding_side != "left":
            raise ValueError(
                "Tokenizer padding side must be 'left' for text generation."
            )

        batches = [
            all_messages[i : i + batch_size]
            for i in range(0, len(all_messages), batch_size)
        ]
        response_texts = []
        responses = []

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
                do_sample=True,  # required for temperature and top_p to work
                temperature=sampling_params.temperature,
                max_new_tokens=sampling_params.max_tokens,
                top_p=sampling_params.top_p,
                **generate_kwargs,
            )

            # AutoModelForCausalLM does not allow to return only the generated text so manually remove the input
            batch_outputs = batch_outputs[:, batch_inputs.input_ids.shape[1] :]
            batch_texts = tokenizer.batch_decode(
                batch_outputs, skip_special_tokens=True
            )

            response_texts.extend(batch_texts)
            responses.extend(batch_outputs)

    elif isinstance(model, vllm.LLM):
        # * vLLM performs batching internally
        try:
            responses = model.chat(
                all_messages,
                sampling_params=sampling_params,
                chat_template=tokenizer.chat_template,
                # use_tqdm=False, # to avoid spamming the console with progress bars
                # disable thinking for now
                chat_template_kwargs={"enable_thinking": False},
                **generate_kwargs,
            )
        except Exception as e:
            print(
                f"Failed to generate responses with vLLM: {e}\nRetrying without fixed chat template..."
            )
            responses = model.chat(
                all_messages,
                sampling_params=sampling_params,
                # use_tqdm=False, # to avoid spamming the console with progress bars
                # disable thinking for now
                chat_template_kwargs={"enable_thinking": False},
                **generate_kwargs,
            )
        response_texts = [_.outputs[0].text for _ in responses]

    elif isinstance(model, Pipeline):
        batches = [
            all_messages[i : i + batch_size]
            for i in range(0, len(all_messages), batch_size)
        ]
        responses = []

        for batch in tqdm(batches, desc="Generating responses", total=len(batches)):
            batch_outputs = model(
                batch,
                return_full_text=False,
                num_return_sequences=1,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                max_new_tokens=sampling_params.max_tokens,
                **generate_kwargs,
            )

            responses.extend(batch_outputs)
        response_texts = [response[0]["generated_text"] for response in responses]
    else:
        raise ValueError(
            f"Was not able to resolve model to be used for generation. model: {model}"
        )

    return response_texts, responses


if __name__ == "__main__":
    setup(login_to_hf=True)
    set_seed(42)
