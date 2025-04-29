import subprocess
import sys
import datasets
from datasets import load_dataset
import argparse


possible_splits = ["test_gen", "test_hard", "chat", "chat_gen"]


def run_rewardbench(model, dataset, split, chat_template, batch_size):
    command = [
        "rewardbench",
        f"--model={model}",
        f"--dataset={dataset}",
        f"--split={split}",
        f"--chat_template={chat_template}",
        f"--batch_size={batch_size}"
    ]
    
    
    try:
        subprocess.run(command, check=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
    except subprocess.CalledProcessError as e:
        print("Command failed with return code", e.returncode)

### Debugging

def debug_rewardbench(model, dataset, split, chat_template, batch_size):
    from rewardbench.rewardbench import main
    sys.argv = [
        "rewardbench",
        f"--model={model}",
        f"--dataset={dataset}",
        f"--split={split}",
        f"--chat_template={chat_template}",
        f"--batch_size={batch_size}"
    ]
    print(sys.argv)
    main()



def custom_reward_evaluation():
    # Load full RewardBench dataset
    dataset = load_dataset("allenai/reward-bench", split="raw")

    # Create mapping from subset to category
    subset_to_category = {
        "AlpacaEval Easy": "Chat",
        "AlpacaEval Length": "Chat",
        "AlpacaEval Hard": "Chat",
        "MT Bench Easy": "Chat",
        "MT Bench Medium": "Chat",
        "MT Bench Hard": "Chat Hard",
        "LLMBar Natural": "Chat Hard",
        "LLMBar Adver. Neighbor": "Chat Hard",
        "LLMBar Adver. GPTInst": "Chat Hard",
        "LLMBar Adver. GPTOUt": "Chat Hard",
        "LLMBar Adver. Manual": "Chat Hard",
        "Refusals Dangerous": "Safety",
        "Refusals Offensive": "Safety",
        "XSTest Should Refuse": "Safety",
        "XSTest Should Respond": "Safety",
        "Do Not Answer": "Safety",
        "PRM Math": "Reasoning",
        "HumanEvalPack CPP": "Reasoning",
        "HumanEvalPack Go": "Reasoning",
        "HumanEvalPack Javascript": "Reasoning",
        "HumanEvalPack Java": "Reasoning",
        "HumanEvalPack Python": "Reasoning",
        "HumanEvalPack Rust": "Reasoning",
        "Anthropic Helpful": "Prior Sets",
        "Anthropic HHH": "Prior Sets",
        "SHP": "Prior Sets",
        "Summarize": "Prior Sets",
    }

    # Add category to each example
    dataset = dataset.map(lambda x: {"category": subset_to_category.get(x["subset"], "Unknown")})

    # Now you can filter
    chat_examples = dataset.filter(lambda x: x["category"] == "Chat")
    safety_examples = dataset.filter(lambda x: x["category"] == "Safety")
    reasoning_examples = dataset.filter(lambda x: x["category"] == "Reasoning")


#reward4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train reward model using reward config YAML.")
    parser.add_argument("--model", required=True, help="Path to model config or default model like OpenAssistant/reward-model-deberta-v3-base.")
    parser.add_argument("--dataset", default ="allenai/ultrafeedback_binarized_cleaned", help="Datset rewardbench to be evaluated on.")
    parser.add_argument("--split", default ="test_gen", help="Split of rewardbench dataset.")
    parser.add_argument("--batch-size", default=3, help="Batch Size for reward evaluation.")
    parser.add_argument("--chat-template", default="raw", help="Chat template for reward evaluation.")
    args = parser.parse_args()


    #OpenAssistant/reward-model-deberta-v3-base
    #ds = load_dataset("allenai/ultrafeedback_binarized_cleaned")
    #print(ds.keys())
    run_rewardbench(args.model, args.dataset, args.split, args.chat_template, args.batch_size)
    #custom_reward_evaluation()
    #print("Using dataset:", args.dataset)
    #for split in possible_splits:
    #    run_rewardbench(split)