from activeuf.prompts import *

PUBLIC_ENV_PATH = ".env"
LOCAL_ENV_PATH = ".env.local"

SEED = 123
MAX_NUM_GPUS = 4
MODEL_CLASS = "transformers"  # Which package to use for the model. ["transformers", "pipeline" "vllm"]

# ====================================
#               DATASETS              
# ====================================

DATASET_MAP = {
    "truthful_qa": "truthfulqa/truthful_qa", 
    # "false_qa": "",
    # "sharegpt": "",
    # "ultrachat": "",
    # "flan": "",
    # "evol_instruct": "",
}
DATASET_POOL = list(DATASET_MAP.keys())

# ====================================
#               MODELS              
# ====================================

# TODO: Complete model map
MODEL_MAP = {
    "gemma-3-1b": "google/gemma-3-1b-it",

    "smollm-2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm-2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm-2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",

    "qwen-2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen-2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",

    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",

    "phi-4": "microsoft/phi-4",
    "phi-4-mini": "microsoft/Phi-4-mini-instruct",
}
MODEL_POOL = list(MODEL_MAP.keys())

# TODO: Consolidate model map and chat template somehow, maybe in a new class
# NOTE: UltraLM series does not support the "system" role, see https://huggingface.co/openbmb/UltraLM-13b
GPT2_CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
ULTRALM_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] in ['user', 'system'] %}{{ 'User: ' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] }}\n{% endif %}{% if loop.last and add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}{% endfor %}"
LLAMA_CHAT_TEMPLATE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n" + "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}\n{% elif message['role'] == 'user' %}{{ 'USER: ' + message['content'] }}\n{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + eos_token}}\n{% endif %}{% if loop.last and add_generation_prompt %}{{ 'ASSISTANT: ' }}{% endif %}{% endfor %}"
DEBERTA_CHAT_TEMPLATE = "{% for message in messages %}\n{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}\n"
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\nUser: {{ message['content'] }}\n{% elif message['role'] == 'assistant' %}\nAssistant: {{ message['content'] }}\n{% elif message['role'] == 'system' %}\nSystem: {{ message['content'] }}\n{% endif %}\n{% endfor %}"

# Define custom chat templates only for models that don't already have a default chat template
MODEL2CHAT_TEMPLATE = {
    # ! Small models are only used for debugging, so they do not necessarily need a custom chat template
    "gpt-2": GPT2_CHAT_TEMPLATE,
    # "opt-strict-125m": GPT2_CHAT_TEMPLATE,
    # "babyllama-10m": GPT2_CHAT_TEMPLATE,
    # "babyllama-100m": GPT2_CHAT_TEMPLATE,

    "ultralm-13b": ULTRALM_CHAT_TEMPLATE,
    "ultralm-65b": ULTRALM_CHAT_TEMPLATE,

    "vicuna-7b": LLAMA_CHAT_TEMPLATE,
    "vicuna-13b": LLAMA_CHAT_TEMPLATE,

    "wizardlm-7b": LLAMA_CHAT_TEMPLATE,
    "wizardlm-13b": LLAMA_CHAT_TEMPLATE,
    "wizardlm-70b": LLAMA_CHAT_TEMPLATE,
    
    "deberta-v3-base": DEBERTA_CHAT_TEMPLATE,
}

# Define the data type with which each model should be loaded
MODEL2DTYPE = {
    "starchat": "bfloat16",
    "mpt-30b-chat": "bfloat16",
    "falcon-40b-instruct": "bfloat16",
}

# ====================================
#         COMPLETION GENERATION       
# ====================================

# ! When changing this from 4, the prompt template needs to be changed as well
NUM_MODELS = len(MODEL_POOL)  

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
PRINCIPLE2PROMPTS = {
    "helpfulness": HELPFULNESS_COMPLETION_SYSTEM_PROMPTS,
    "honesty": HONESTY_COMPLETION_SYSTEM_PROMPTS,
    "truthfulness": TRUTHFULNESS_COMPLETION_SYSTEM_PROMPTS,
    "verbalized_calibration": VERBALIZED_CALIBRATION_COMPLETION_SYSTEM_PROMPTS,
}

# Define which principles are used for which datasets
DATASET2PRINCIPLE_POOL = {
    "truthful_qa": ["honesty", "truthfulness"],
    # "sharegpt": ["helpfulness", "honesty", "truthfulness"],
    # "ultrachat": ["helpfulness", "honesty", "truthfulness"],
    # "flan": ["helpfulness", "verbalized_calibration"],
    # "false_qa": ["honesty", "truthfulness"],
    # "evol_instruct": ["helpfulness"],
}

# ====================================
#             ANNOTATION
# ====================================

ANNOTATION_MODEL = "" 

# General parameters for the annotation step
ANNOTATE_PREFERENCE = True
ANNOTATE_CRITIQUE = True
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
    "truthfulness_without_answer": TRUTHFULNESS_WITHOUT_ANSWER_ANNOTATION_SYSTEM_PROMPT,
    "helpfulness": HELPFULNESS_ANNOTATION_SYSTEM_PROMPT,
    "helpfulness_without_answer": HELPFULNESS_WITHOUT_ANSWER_ANNOTATION_SYSTEM_PROMPT,
    "feedback": FEEDBACK_ANNOTATION_SYSTEM_PROMPT
}

# Regex patterns used to extract the ratings and rationales from the annotation model's response
ASPECT2ANNOTATION_PATTERN = {
    "instruction_following": r"Rating: (.+?)\nRationale: (.+)",
    "honesty": r"Rating: (.+?)\nRationale: (.+)",
    "truthfulness": r"Type: (.+?)\nRationale: (.+?)\nRating: (.+?)\nRationale: (.+)",
    "helpfulness": r"Type: (.+?)\nRationale: (.+?)\nRating: (.+?)\nRationale: (.+)"
}

# Sanity checks
assert DEFAULT_PRINCIPLE in PRINCIPLES
assert sorted(list(PRINCIPLE2PROMPTS.keys())) == sorted(PRINCIPLES)
assert sorted(list(DATASET2PRINCIPLE_POOL.keys())) == sorted(DATASET_POOL)
assert set(principle for pool in DATASET2PRINCIPLE_POOL.values() for principle in pool).issubset(set(PRINCIPLES))
# assert set(MODEL2CHAT_TEMPLATE.keys()).issubset(set(MODEL_POOL))