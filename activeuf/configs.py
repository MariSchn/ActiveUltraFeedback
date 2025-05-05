from activeuf.prompts import *

PUBLIC_ENV_PATH = ".env"
LOCAL_ENV_PATH = ".env.local"

SEED = 123
MAX_NUM_GPUS = 4
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

COMPLETION_MODEL_PATHS = {
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

# ! When changing this from 4, the prompt template needs to be changed as well
NUM_COMPLETION_MODELS = len(COMPLETION_MODEL_PATHS)

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
ASPECT2ANNOTATION_PROMPT = {
    "instruction_following": INSTRUCTION_FOLLOWING_ANNOTATION_SYSTEM_PROMPT,
    "honesty": HONESTY_ANNOTATION_SYSTEM_PROMPT,
    "truthfulness": TRUTHFULNESS_ANNOTATION_SYSTEM_PROMPT,
    "helpfulness": HELPFULNESS_ANNOTATION_SYSTEM_PROMPT,
}

# Regex patterns used to extract the ratings and rationales from the annotation model's response
ASPECT2ANNOTATION_PATTERN = {
    "instruction_following": r"Rating: (.+?)\nRationale: (.+)",
    "honesty": r"Rating: (.+?)\nRationale: (.+)",
    "truthfulness": r"Type: (.+?)\nType rationale: (.+?)\nRating: (.+?)\nRationale: (.+)",
    "helpfulness": r"Type: (.+?)\nType rationale: (.+?)\nRating: (.+?)\nRationale: (.+)",
}
FEEDBACK_ANNOTATION_PATTERN = r"Feedback: (.+?)\nOverall score: (\d+)"