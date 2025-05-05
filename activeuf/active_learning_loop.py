import random
import torch
import json
import argparse
import yaml
import numpy as np
from torch.utils.data import DataLoader
from activeuf.uncertainty_quantification.classes import UQTokenizer, UQModelClass, UQTrainer
from activeuf.oracle.classes import Oracle
from activeuf.acquisition_function.acquisition import RandomAcquisitionFunction, DoubleThompsonSampling

def load_prompts_with_completions(completion_dataset, batch_size, shuffle=True) -> DataLoader:
    data = []

    with open(completion_dataset, 'r') as f:
        first_char = f.read(1)
        f.seek(0)  # Reset file pointer

        if first_char == '[':
            # Standard JSON array
            data = json.load(f)
        else:
            # Assume JSON Lines format
            for line in f:
                if line.strip():  # skip empty lines
                    data.append(json.loads(line))

    data = np.array(data)    
    collate_fn = lambda batch: {key: [d[key] for d in batch] for key in batch[0].keys()}

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    return dataloader


def prompts_with_two_completions(prompts_with_completions, rewards, uncertainty, acquisition_function):
    # An array consisting of 2-element arrays (chosen completion indices)
    selected_ids_per_prompt = acquisition_function(rewards, uncertainty)
    prompts_with_two_completions_for_annotation = []

    for idx, selected_ids in enumerate(selected_ids_per_prompt):
        prompt = prompts_with_completions["instruction"][idx]
        completions = prompts_with_completions['completions'][idx]
        
        selected_completion1 = completions[selected_ids[0]]
        selected_completion2 = completions[selected_ids[1]]

        prompts_with_two_completions_for_annotation.append({
            "prompt": prompt,
            "selected1": selected_completion1,
            "selected2": selected_completion2,
        })

    return prompts_with_two_completions_for_annotation


def save_ultrafeedback_format(prompts_with_completions_for_annotation, output_path):

    # Save to .jsonl
    with open(output_path, 'w') as f:
        for prompt in prompts_with_completions_for_annotation:
            f.write(json.dumps(prompt) + '\n')

    print(f"✅ Saved {len(prompts_with_completions_for_annotation)} records to {output_path}")
    return prompts_with_completions_for_annotation    


def uncertainty_sampling_loop(uq_model_path, uq_model_config, uq_trainer_path, completion_dataset, num_iterations, acquisition_function_type, batch_size, acquisition_config):
    uq_tokenizer = UQTokenizer.from_pretrained(uq_model_path, uq_model_config)
    uq_model = UQModelClass.from_pretrained(uq_model_path)
    uq_trainer = UQTrainer(UQTokenizer, uq_model, uq_trainer_path)

    dataset = []
    oracle = Oracle()
    
    if acquisition_function_type == "double_thompson_sampling":
        max_iterations = acquisition_config.get("max_iterations", 10)
        beta = acquisition_config.get("beta", 1)
        acquisition_function = DoubleThompsonSampling(beta=beta, max_iterations=max_iterations) # will be changed later.
    else: 
        acquisition_function = RandomAcquisitionFunction()

    prompts_with_completions_dataloader = load_prompts_with_completions(completion_dataset, batch_size)
    
    iteration_number = 0
    for prompts_with_completions in prompts_with_completions_dataloader:
        inputs = uq_tokenizer(prompts_with_completions)  # Dict[str, torch.Tensor] - Dictionary with keys “input_ids” and “attention_maks”
        rewards, uncertainty = uq_trainer.model(inputs)  # torch.Tensors with Shape: (n_prompts, n_completions)
        prompts_with_completions_for_annotation = prompts_with_two_completions(prompts_with_completions, rewards, uncertainty, acquisition_function)
        labels = oracle(prompts_with_completions_for_annotation)
        dataset.append((prompts_with_completions_for_annotation, labels))
        uq_trainer.training_step(prompts_with_completions_for_annotation, labels)
        if iteration_number == num_iterations - 1: break 
        else: iteration_number += 1
        

    #TODO: save dataset, Martin's job. (We can add prompt_ids here)
    
def main(config):
    try:
        # Attempt to load the reward configuration file
        with open(config.acquisition_config, "r") as f:
            acquisition_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: The specified reward configuration file '{config.acquisition_config}' was not found.")
        print("Continuing with default parameters")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse the reward configuration file '{config.reward_config}'.")
        print(f"Details: {e}")
        print("Continuing with default parameters")
    except Exception as e:
        print(f"An unexpected error occurred while loading the reward configuration file: {e}")
        print("Continuing with default parameters")

    uncertainty_sampling_loop(config.uq_model_path, config.uq_model_config, config.uq_trainer_path, config.completion_dataset, config.num_iterations, config.acquisition_function_type, config.batch_size, acquisition_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train reward model using reward config YAML.")
    parser.add_argument("--uq-model-path", type=str, help="Path to uncertainty quantification model.")
    parser.add_argument("--uq-model-config", type=str, help="Path to uncertainty quantification config.")
    parser.add_argument("--uq-trainer-path", type=str, help="Path to uncertainty quantification trainer.")
    parser.add_argument("--completion-dataset", type=str, required=True, help="Path to the prompt dataset.")
    parser.add_argument("--num-iterations", type=int, default=10, help="Number of iterations in uncertainty sampling.")
    parser.add_argument("--batch-size", type=int, default=3, help="Batch Size for uncertainty sampling.")
    parser.add_argument("--acquisition_function_type", type=str, default="double_thompson_sampling", help="Acquistion function type")
    parser.add_argument("--acquisition_config", type=str, default="activeuf/acquisition_function/acquisition_config.yaml", help="acquisition function configuration file path")
    config = parser.parse_args()

    main(config)