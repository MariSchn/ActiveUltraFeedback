import random
import torch
import json
import argparse
import numpy as np
from activeuf.uncertainty_quantification.classes import UQTokenizer, UQModelClass, UQTrainer
from activeuf.oracle.classes import Oracle
from activeuf.aqcuisition_function.aqcuisition import RandomAcquisitionFunction

def load_prompts_with_completions(completion_dataset):
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
    return np.array(data)


def select_prompts_with_completions(acquisition_function, prompts_with_completions, rewards, uncertainty):
    selected_triplets = acquisition_function.select(rewards, uncertainty)
    prompts_with_completions_for_annotation = []

    for prompt_idx, chosen_idx, rejected_idx in selected_triplets:
        prompt = prompts_with_completions[prompt_idx]
        completions = prompt['completions']
        prompt_rewards = rewards[prompt_idx]

        chosen = completions[chosen_idx]
        rejected = completions[rejected_idx]

        prompts_with_completions_for_annotation.append({
            "prompt": prompt['instruction'],
            "prompt_id": f"prompt_{prompt_idx}",
            "chosen": chosen['response_text'],
            "rejected": rejected['response_text'],
            "score_chosen": chosen.get('reward', prompt_rewards[chosen_idx]),
            "score_rejected": rejected.get('reward', prompt_rewards[rejected_idx]),
            "source": "---"
        })
    return prompts_with_completions_for_annotation


def save_ultrafeedback_format(prompts_with_completions_for_annotation, output_path):

    # Save to .jsonl
    with open(output_path, 'w') as f:
        for prompt in prompts_with_completions_for_annotation:
            f.write(json.dumps(prompt) + '\n')

    print(f"✅ Saved {len(prompts_with_completions_for_annotation)} records to {output_path}")
    return prompts_with_completions_for_annotation    


def uncertainty_sampling_loop(uq_model_path, uq_model_config, uq_trainer_path, completion_dataset, num_iterations, aqcuisition_function_type, batch_size):
    uq_tokenizer = UQTokenizer.from_pretrained(uq_model_path, uq_model_config)
    uq_model = UQModelClass.from_pretrained(uq_model_path)
    uq_trainer = UQTrainer(UQTokenizer, uq_model, uq_trainer_path)

    dataset = []
    oracle = Oracle()
    
    if aqcuisition_function_type == "double_thompson_sampling":
        acquisition_function = RandomAcquisitionFunction() # will be changed later.
    else: 
        acquisition_function = RandomAcquisitionFunction()

    for i in range(num_iterations):
        prompts_with_completions = load_prompts_with_completions(completion_dataset)
        inputs = uq_tokenizer(prompts_with_completions)  # Dict[str, torch.Tensor] - Dictionary with keys “input_ids” and “attention_maks”
        rewards, uncertainty = uq_trainer.model(inputs)  # torch.Tensors with Shape: (n_prompts, n_completions)
        prompts_with_completions_for_annotation = select_prompts_with_completions(acquisition_function, prompts_with_completions,rewards, uncertainty)
        labels = oracle(prompts_with_completions_for_annotation)
        dataset.append((prompts_with_completions_for_annotation, labels))
        uq_trainer.training_step(prompts_with_completions_for_annotation, labels)

    #TODO: save dataset, Martin's job.
    
def main(config):
    uncertainty_sampling_loop(config.uq_model_path, config.uq_model_config, config.uq_trainer_path, config.completion_dataset, config.num_iterations, config.aqcuisition_function_type, config.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train reward model using reward config YAML.")
    parser.add_argument("--uq-model-path", type=str, help="Path to uncertainty quantification model.")
    parser.add_argument("--uq-model-config", type=str, help="Path to uncertainty quantification config.")
    parser.add_argument("--uq-trainer-path", type=str, help="Path to uncertainty quantification trainer.")
    parser.add_argument("--completion-dataset", type=str, required=True, help="Number of iterations in uncertainty sampling.")
    parser.add_argument("--num-iterations", type=int, default=10, help="Number of iterations in uncertainty sampling.")
    parser.add_argument("--batch-size", type=int, default=3, help="Batch Size for uncertainty sampling.")
    parser.add_argument("--aqcuisition_function_type", type=str, help="Acquistion function type")
    config = parser.parse_args()

    main(config)