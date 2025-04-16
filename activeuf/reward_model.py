from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RewardModelInference:
    def __init__(self, model_filename):
        """
        Initialize the RewardModel by loading the model and tokenizer.

        Args:
            model_filename (str): Path to the pretrained model directory or model name.
        """
        self.model_filename = model_filename
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_filename)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_filename)
            print(self.model.config)
            #print(self.tokenizer.config)
        except:
            raise RuntimeError(f"model or tokenizer on path {model_filename} was not found...")

    def reward(self, formatted_input):
        """
        Calculate the reward for a given formatted input.

        Args:
            formatted_input:
                Reward model expects a formatted input.

        Returns:
            float: The reward score.
        """
        # Tokenize the input
        inputs = self.tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True)

        # Perform inference
        outputs = self.model(**inputs)

        # Extract the reward score (logits)
        reward_score = outputs.logits.item()
        return reward_score
    

"""
example:
"""
model = RewardModelInference("reward8")
message = "user: What is the capital of France?\nassistant: The capital of France is Paris.\n"
print(model.reward(message))

