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

    model = LLM(model="Qwen/Qwen2.5-72B-Instruct", tensor_parallel_size=4)

    prompt = "What is the capital of France?"

    response = model.generate(prompt)
    print(response)