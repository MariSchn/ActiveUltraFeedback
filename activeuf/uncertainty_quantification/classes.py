import random


def calc_reward_uncertainty_random(completion_data):
    completion_data = completion_data['input_ids']

    rewards_list = []        # List of lists: rewards per prompt
    uncertainties_list = []  # List of lists: uncertainties per prompt

    for entry in completion_data:
        prompt_rewards = []
        prompt_uncertainties = []

        for comp in entry.get('completions', []):
            reward = round(random.uniform(-1.0, 1.0), 3)
            uncertainty = round(random.uniform(0.0, 1.0), 3)

            comp['reward'] = reward
            comp['uncertainty'] = uncertainty

            prompt_rewards.append(reward)
            prompt_uncertainties.append(uncertainty)

        rewards_list.append(prompt_rewards)
        uncertainties_list.append(prompt_uncertainties)

    return rewards_list, uncertainties_list


class UQTokenizer:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, path, config=None):
        # Simulate loading from a path/config
        print(f"Loading tokenizer from {path} with config {config}")
        return cls()

    def __call__(self, prompts_with_completions):
        # Tokenize here; for now just simulate output
        return {"input_ids": prompts_with_completions, "attention_mask": [1] * len(prompts_with_completions)}


class UQModelClass:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, path, config=None):
        # Simulate loading from a path/config
        print(f"Loading model from {path} with config {config}")
        return cls()

    def __call__(self, tokenized_prompts_with_completions):
        rewards, uncertainty = calc_reward_uncertainty_random(tokenized_prompts_with_completions)
        return rewards, uncertainty



class UQTrainer():

    def __init__(self, tokenizer, model, trainer_config):
        self.tokenizer = tokenizer
        self.model = model
        self.trainer_config = trainer_config


    def training_step(self, prompts_with_completions_for_annotation, labels):
        pass