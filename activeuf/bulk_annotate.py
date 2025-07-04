import argparse
import os.path as path

from datasets import load_from_disk

from activeuf.configs import *
from activeuf.prompts import *
from activeuf.schemas import *
from activeuf.utils import *

"""
python -m activeuf.bulk_annotate \
    --inputs_path datasets/inputs_for_bulk_annotation \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --download_dir ./hf_cache \
    --part_size 30000 \
    --part 0
"""

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs_path", type=str, required=True, help="The path to the inputs for bulk annotation")
    parser.add_argument("--output_path", type=str, help="Where to export the bulk annotations")

    parser.add_argument("--model_name", type=str, required=True, help="The Huggingface path or API of the model to use for annotations (e.g. HuggingFaceTB/SmolLM2-135M-Instruct, gpt-4)")
    parser.add_argument("--model_class", type=str, default=DEFAULT_MODEL_CLASS, help="The class which is used to perform inference (e.g. transformers, pipeline, vllm)")
    parser.add_argument("--download_dir", type=str, help="The path to the Huggingface cache directory. If not set, the default Huggingface cache directory is used.")

    parser.add_argument("--max_tokens", type=int, default=ANNOTATION_MAX_TOKENS, help="The maximum number of tokens for LLM responses")
    parser.add_argument("--temperature", type=float, default=ANNOTATION_TEMPERATURE, help="The temperature for sampling")
    parser.add_argument("--top_p", type=float, default=ANNOTATION_TOP_P, help="The top_p for sampling")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for random sampling")
    parser.add_argument("--max_num_gpus", type=int, default=MAX_NUM_GPUS, help="The maximum number of GPUs to use")

    parser.add_argument("--part_size", type=int, default=None, help="If set, will perform only part_size many annotations")
    parser.add_argument("--part", type=int, default=None, help="If set, will split the dataset into parts of size part_size, and only annotate the part at this index")

    parser.add_argument("--debug", action="store_true", help="If set, will only annotate the first few samples")
    args = parser.parse_args()

    if not args.output_path:
        args.output_path = f"datasets/bulk_annotations"
        if args.part is not None:
            args.output_path += f"/part_{args.part}"
    assert not path.exists(args.output_path), f"Output path {args.output_path} already exists"

    return args

if __name__ == "__main__":
    args = parse_args()

    logger.info("Logging into HuggingFace")
    setup(login_to_hf=True)

    if args.seed:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)
        
    logger.info(f"Loading {args.inputs_path}")
    dataset = load_from_disk(args.inputs_path)
    if args.part is not None and args.part_size is not None:
        logger.info(f"Annotating just part {args.part} (size {args.part}) of the inputs given")
        dataset = dataset.select(
            range(args.part_size*args.part, args.part_size*(args.part+1))
        )
    if args.debug:
        logger.info("Debug mode: only annotating the first few inputs")
        dataset = dataset.select(range(10))
        
    logger.info(f"Using {args.model_name} for annotation")
    if args.download_dir:
        model, tokenizer = load_model(args.model_name, args.model_class, download_dir=args.download_dir)
    else:
        model, tokenizer = load_model(args.model_name, args.model_class)
    sampling_params = SamplingParams(
        max_tokens = args.max_tokens,
        temperature = args.temperature,
        top_p = args.top_p,      
    )

    raw_annotations = get_response_texts(model, tokenizer, dataset["messages"], sampling_params)
    dataset = dataset.add_column("raw_annotations", raw_annotations)

    dataset.save_to_disk(args.output_path)