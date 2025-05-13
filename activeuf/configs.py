from activeuf.prompts import *

PUBLIC_ENV_PATH = ".env"
LOCAL_ENV_PATH = ".env.local"

SEED = 123
MAX_NUM_GPUS = 4
MAX_API_RETRY = 10
DEFAULT_MODEL_CLASS = "vllm"  # Which package to use for the model. ["transformers", "pipeline" "vllm"]

# ====================================
#               DATASETS              
# ====================================

PROMPT_SOURCES = {
    "evol_instruct",
    "false_qa",
    "flan_v2_cot",
    "flan_v2_flan2021",
    "flan_v2_niv2",
    "flan_v2_p3",
    "sharegpt",
    "ultrachat",
}

# ====================================
#         COMPLETION GENERATION       
# ====================================

MODEL_APIS = {
    "gpt-3",
    "gpt-4",
}

COMPLETION_MODEL_NAMES = {
    "google/gemma-3-1b-it",

    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",

    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",

    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",

    "microsoft/phi-4",
    "microsoft/Phi-4-mini-instruct",
}
NUM_COMPLETION_MODELS = len(COMPLETION_MODEL_NAMES)

# General parameters for the completions generation step
COMPLETION_MAX_TOKENS = 1024
COMPLETION_TEMPERATURE = 1.0
COMPLETION_TOP_P = 1.0

# Principles to be used in the system prompt when generating completions
PRINCIPLES = [
    "helpfulness",
    "honesty",
    "truthfulness",
    "verbalized_calibration"
]
DEFAULT_PRINCIPLE = "helpfulness"

# System prompts to be used when generating completions
PRINCIPLE2SYSTEM_PROMPTS = {
    "helpfulness": HELPFULNESS_COMPLETION_SYSTEM_PROMPTS,
    "honesty": HONESTY_COMPLETION_SYSTEM_PROMPTS,
    "truthfulness": TRUTHFULNESS_COMPLETION_SYSTEM_PROMPTS,
    "verbalized_calibration": VERBALIZED_CALIBRATION_COMPLETION_SYSTEM_PROMPTS,
}

# Define which principles are used for which datasets
PROMPT_SOURCE2PRINCIPLES = {
    "truthful_qa": ["honesty", "truthfulness"],
    "sharegpt": ["helpfulness", "honesty", "truthfulness"],
    "ultrachat": ["helpfulness", "honesty", "truthfulness"],
    "flan": ["helpfulness", "verbalized_calibration"],
    "false_qa": ["honesty", "truthfulness"],
    "evol_instruct": ["helpfulness"],
}

# ====================================
#             ANNOTATION
# ====================================

# Model to use for annotating completions
ANNOTATION_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct" 

# General parameters for the annotation step
ANNOTATION_MAX_TOKENS = 10
ANNOTATION_TEMPERATURE = 1.0
ANNOTATION_TOP_P = 1.0
