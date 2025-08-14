from activeuf.prompts import *

PUBLIC_ENV_PATH = ".env"
LOCAL_ENV_PATH = ".env.local"

LOGS_DIR = "logs"

SEED = 123
MAX_NUM_GPUS = 4
MAX_NUM_NODES = 1
MAX_API_RETRY = 10
# Which package to use for the model. ["transformers", "pipeline" "vllm", "vllm_server"]
DEFAULT_MODEL_CLASS = "vllm"

VLLM_SERVER_BASE_URL = "http://localhost:8000"  # URL of the vLLM server
PING_DELAY = 10        # Delay between pings to the server to check if it is already running
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
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",  # 402B (17B Active)

    "Qwen/Qwen3-30B-A3B",                            # 30B  (03B Active)
    "Qwen/Qwen3-235B-A22B",                          # 235B (22B Active)

    "deepseek-ai/DeepSeek-V3",                       # 671B (37B Active)

    "moonshotai/Moonlight-16B-A3B-Instruct",         # 16B (03B Active)
    "moonshotai/Kimi-K2-Instruct",                   # 1000B (32B Active)
}

# ! When changing this from 4, the prompt template needs to be changed as well
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

ANNOTATION_MODEL = ""

# General parameters for the annotation step
NUM_SHUFFLES = 1

ANNOTATION_MAX_TOKENS = 1024
ANNOTATION_TEMPERATURE = 1.0
ANNOTATION_TOP_P = 1.0

# How often to retry calling the API for models that require API calls.
MAX_API_RETRY = 10
# How often to retry parsing the response from the annotating model. This might fail as the model is not always guaranteed to follow the expected format.
# Keep in mind that trying to parse again requires to re-run the model again, which can be expensive (O(MAX_API_RETRY * MAX_PARSE_RETRY)).
MAX_PARSE_RETRY = 10

# Aspects to be used to annotate the generated completions
ASPECTS = [
    "instruction_following",
    "helpfulness",
    "honesty",
    "truthfulness"
]

# Map an aspect to the corresponding system prompt (template) that is used to annotate the generated completions
# ASPECT2ANNOTATION_PROMPT = {
#     "instruction_following": INSTRUCTION_FOLLOWING_ANNOTATION_SYSTEM_PROMPT,
#     "honesty": HONESTY_ANNOTATION_SYSTEM_PROMPT,
#     "truthfulness": TRUTHFULNESS_ANNOTATION_SYSTEM_PROMPT,
#     "helpfulness": HELPFULNESS_ANNOTATION_SYSTEM_PROMPT,
# }

# Regex patterns used to extract the ratings and rationales from the annotation model's response
ASPECT2ANNOTATION_PATTERN = {
    "instruction_following": r"Rating:(.+?)Rationale:(.+)",
    "honesty": r"Rating:(.+?)Rationale:(.+)",
    "truthfulness": r"Type:(.+?)Type rationale:(.+?)Rating:(.+?)Rationale:(.+)",
    "helpfulness": r"Type:(.+?)Type rationale:(.+?)Rating:(.+?)Rationale:(.+)",
}
FEEDBACK_ANNOTATION_PATTERN = r"Feedback:(.+?)Overall score:(.+)"
