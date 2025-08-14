import torch
from vllm import LLM
from vllm.utils import is_torch_equal_or_newer
from activeuf.utils import setup


if __name__ == "__main__":
    setup(login_to_hf=True)

    print(f"PyTorch version: {torch.__version__}")
    print(f"NCCL version: {torch.cuda.nccl.version()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"is_torch_equal_or_newer(2.6): {is_torch_equal_or_newer('2.6')}")
    print(f"is_torch_equal_or_newer(2.6.0): {is_torch_equal_or_newer('2.6.0')}")
    print(f"torch.__version__>=2.6: {torch.__version__ >= '2.6'}")
    print(f"torch.__version__>=2.6.0: {torch.__version__ >= '2.6.0'}")
    print(f"str(torch.__version__)>=2.6: {torch.__version__ >= '2.6'}")
    print(f"str(torch.__version__)>=2.6.0: {torch.__version__ >= '2.6.0'}")

    model = LLM(model="meta-llama/Llama-3.2-3B", download_dir="/iopsstor/scratch/cscs/smarian/hf_cache", enforce_eager=True)  # This will disable torch.compile)

    prompt = "What is the capital of France?"

    response = model.generate(prompt)
    print(response)