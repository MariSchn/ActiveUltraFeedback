import argparse
import json
from tqdm import tqdm

from vllm import LLM

from .schemas import Sample
from .configs import MAX_NUM_GPUS
from .utils import set_seed

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_path", type=str, required=True, help="The path to the input dataset")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use for completions (e.g. llama-2-13b-chat)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="The maximum number of tokens to generate for each completion")
    parser.add_argument("--seed", type=int, default=123, help="Seed for random sampling")

    parser.add_argument("--output_dir", type=str, default="./datasets_with_completions/", help="The directory for exporting the generated completions")
    return parser.parse_args()

def load_input_data(input_dataset_path: str) -> Sample:
    with open(input_dataset_path, "r") as f_in:
        for line in tqdm(f_in):
            yield Sample(**json.loads(line))

def load_model(model_name: str) -> LLM:
    if model_name in ["starchat", "mpt-30b-chat", "falcon-40b-instruct"]:
        dtype = "bfloat16"
    else:
        dtype = "auto"

    return LLM(
        model_name, 
        gpu_memory_utilization=0.95, 
        swap_space=1, 
        tensor_parallel_size=min(MAX_NUM_GPUS, torch.cuda.device_count()), 
        trust_remote_code=True, 
        dtype=dtype,
    )

if __name__ == "__main__":
    args = parse_args()

    # Set random seed
    if isinstance(args.seed, int):
        set_seed(args.seed)

    # Load generation model
    model = load_model(args.model_name)

    # For each sample
    for sample in load_input_data(args.input_dataset_path):
        if args.model_name not in sample.model_names:
            pass
        
        