import numpy as np
import os
import random
import torch

from vllm import LLM

from activeuf.configs import PRINCIPLES, DEFAULT_PRINCIPLE, DATASET2PRINCIPLE_POOL, MODEL_MAP

from activeuf.comparison_data_generation.fastchat import conv_template

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def sample_principle_for_dataset(dataset_name: str) -> str:
    principle_pool = DATASET2PRINCIPLE_POOL.get(dataset_name, [DEFAULT_PRINCIPLE])
    principle = random.choice(principle_pool)

    if principle == "honesty":
        if "verbalized_calibration" in PRINCIPLES and np.random.rand() < 0.9:
            principle = "verbalized_calibration"

    return principle

def get_stop_tokens(model_name: str, model: LLM = None) -> list[str]:
    # TODO: make the templating nicer and check the stop tokens
    if model_name.split("-")[0] in ["llama", "alpaca", "vicuna", "mpt", "falcon", "wizardlm"]:
        conv = conv_template[model_name.split("-")[0]].copy()
        if conv.stop_str is not None:
            return [conv.stop_str]
        elif conv.stop_token_ids is not None:
            return [model.llm_engine.tokenizer.decode(stop_token_id) for stop_token_id in conv.stop_token_ids]
        else:
            return ["</s>"]
    else:
        return ["</s>"]
    
def load_model(model_name: str, max_num_gpus: int = None) -> LLM:
    # get HF model path
    model_path = MODEL_MAP[model_name]

    # determine model params
    tensor_parallel_size = torch.cuda.device_count()
    if isinstance(max_num_gpus, int):
        tensor_parallel_size = min(max_num_gpus, tensor_parallel_size)

    if model_name in ["starchat", "mpt-30b-chat", "falcon-40b-instruct"]:
        dtype = "bfloat16"
    else:
        dtype = "auto"

    return LLM(
        model_path, 
        gpu_memory_utilization=0.95, 
        swap_space=1, 
        tensor_parallel_size=tensor_parallel_size, 
        trust_remote_code=True, 
        dtype=dtype,
    )
