from activeuf.prompts import *

PUBLIC_ENV_PATH = ".env"
LOCAL_ENV_PATH = ".env.local"

LOGS_DIR = "logs"

SEED = 123
MAX_NUM_GPUS = 4
MAX_NUM_NODES = 1
DATA_PARALLEL_SIZE = 1  # Only compatible with "vllm_server" model class
MAX_API_RETRY = 10
DEFAULT_MODEL_CLASS = "vllm"  # Which package to use for the model. ["transformers", "pipeline" "vllm", "vllm_server"]

VLLM_SERVER_BASE_URL = "http://localhost:8000"  # URL of the vLLM server
PING_DELAY = 30        # Delay between pings to the server to check if it is already running
MAX_PING_RETRIES = 30  # Number of retries to check if the server is running

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
    # ===== DENSE MODELS =====
    "Qwen/Qwen2.5-72B-Instruct",                     # 72B

    "Qwen/Qwen3-14B",                                # 14B
    "Qwen/Qwen3-32B",                                # 32B

    "meta-llama/Llama-3.1-8B-Instruct",              # 08B
    "meta-llama/Llama-3.3-70B-Instruct",             # 70B

    "microsoft/phi-4",                               # 14B

    "mistralai/Mistral-Large-Instruct-2411",         # 123B
    "mistralai/Mistral-Small-24B-Instruct-2501",     # 23B

    "google/gemma-3-12b-it",                         # 12B
    "google/gemma-3-27b-it",                         # 27B

    "nvidia/Llama-3_3-Nemotron-Super-49B-v1",        # 49B
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",     # 70B
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",       # 253B

    "CohereLabs/c4ai-command-a-03-2025",             # 111B

    "allenai/OLMo-2-0325-32B-Instruct",              # 32B
    "allenai/Llama-3.1-Tulu-3-70B",                  # 70B
    "allenai/Llama-3.1-Tulu-3-405B",                 # 405B

    # ===== MoE MODELS =====
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",     # 109B (17B Active)
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct", # 402B (17B Active)

    "Qwen/Qwen3-30B-A3B",                            # 30B  (03B Active)
    "Qwen/Qwen3-235B-A22B",                          # 235B (22B Active)

    "deepseek-ai/DeepSeek-V3",                       # 671B (37B Active)

    "moonshotai/Moonlight-16B-A3B-Instruct",         # 16B (03B Active)
    "moonshotai/Kimi-K2-Instruct",                   # 1000B (32B Active)
}
NUM_COMPLETION_MODELS = len(COMPLETION_MODEL_NAMES)

# General parameters for the completions generation step
COMPLETION_MAX_TOKENS = 4096
COMPLETION_TEMPERATURE = 1.0
COMPLETION_TOP_P = 1.0

# Principles to be used in the system prompt when generating completions
PRINCIPLES = [
    "helpfulness",
    "honesty",
    "truthfulness",
    "persona"
]
DEFAULT_PRINCIPLE = "helpfulness"

# System prompts to be used when generating completions
PRINCIPLE2SYSTEM_PROMPTS = {
    "helpfulness": HELPFULNESS_COMPLETION_SYSTEM_PROMPTS,
    "honesty": HONESTY_COMPLETION_SYSTEM_PROMPTS,
    "truthfulness": TRUTHFULNESS_COMPLETION_SYSTEM_PROMPTS,
    "verbalized_calibration": VERBALIZED_CALIBRATION_COMPLETION_SYSTEM_PROMPTS,
    "persona": PERSONA_COMPLETION_SYSTEM_PROMPTS
}

# Define which principles are used for which datasets
PROMPT_SOURCE2PRINCIPLES = {
    "truthful_qa": ["honesty", "truthfulness"],
    "sharegpt": ["helpfulness", "honesty", "truthfulness"],
    "ultrachat": ["helpfulness", "honesty", "truthfulness"],
    "flan": ["helpfulness", "verbalized_calibration"],
    "false_qa": ["honesty", "truthfulness"],
    "evol_instruct": ["helpfulness"],
    "olmo-2-0325-32b-preference-mix": ["helpfulness", "honesty", "truthfulness"],
    "prism-alignment": ["persona"],
    "PolygloToxicityPrompts_wildchat-id": ["persona"],
    "PolygloToxicityPrompts_wildchat-cs": ["persona"],
    "PolygloToxicityPrompts_wildchat-sv": ["persona"],
    "PolygloToxicityPrompts_wildchat-zh": ["persona"],
    "PolygloToxicityPrompts_wildchat-ko": ["persona"],
    "PolygloToxicityPrompts_wildchat-nl": ["persona"],
    "PolygloToxicityPrompts_wildchat-it": ["persona"],
    "PolygloToxicityPrompts_wildchat-fr": ["persona"],
    "PolygloToxicityPrompts_wildchat-ru": ["persona"],
    "PolygloToxicityPrompts_wildchat-de": ["persona"],
    "PolygloToxicityPrompts_wildchat-pt": ["persona"],
    "PolygloToxicityPrompts_wildchat-ja": ["persona"],
    "PolygloToxicityPrompts_wildchat-ar": ["persona"],
    "PolygloToxicityPrompts_wildchat-en": ["persona"],
    "PolygloToxicityPrompts_wildchat-pl": ["persona"],
    "PolygloToxicityPrompts_wildchat-es": ["persona"],
    "PolygloToxicityPrompts_wildchat-hi": ["persona"],
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
