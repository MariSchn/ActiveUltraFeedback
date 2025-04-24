import argparse
import os

from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from activeuf.dpo.utils import prepare_splits
from activeuf.configs import *
from activeuf.utils import *

"""
This script is designed to train a (SFT) model using direct preference optimization (DPO).
To train the DPO model a (previously SFT trained) model is required and a binarized dataset.
The binarized dataset is a dataset that has been preprocessed to contain pairs of examples, where one example is preferred over the other.

Example run command:
    `python -m activeuf.training --model_path Qwen/Qwen2-0.5B-Instruct --dataset_path trl-lib/ultrafeedback_binarized --debug`

TODO: Add (WandB) logging
TODO: Add config file to avoid too many command line arguments
TODO: Add more customizable (hyper)parameters (e.g. learning rate, batch size, etc.). Only do after config file is added.
TODO: Add better support for models who do not have a chat template defined
"""

def parse_args() -> argparse.Namespace:
    # Parse arguments
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train_path", type=str, help="The path to the local dataset to use for training (e.g. trl-lib/ultrafeedback_binarized/train)")
    parser.add_argument("--test_path", type=str, help="The path to the local dataset to use for testing (e.g. data/trl-lib/ultrafeedback_binarized/test)")
    parser.add_argument("--dataset_path", type=str, help="The HuggingFace path of the dataset to use for training and testing (e.g. trl-lib/ultrafeedback_binarized)")

    parser.add_argument("--model_path", type=str, help="The HuggingFace path of the model to train (e.g. Qwen/Qwen2-0.5B-Instruct)")

    parser.add_argument("--seed", type=int, default=123, help="Seed for random sampling")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode (just one train step on a small subset of the dataset)")

    parser.add_argument("--output_dir", type=str, default="models/dpo/", help="Where to export the tokenizer, trained model, and training args")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Set random seed
    if args.seed:
        set_seed(args.seed)

    # Load and preprocess the dataset splits
    train_split, test_split = prepare_splits(args.dataset_path)
    if args.debug:
        train_split = train_split.select(range(10))
        if test_split:
            val_dataset = test_split.select(range(2))

    # Load model and tokenizer from HF
    # TODO? Support for local model and tokenizer?
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Ensure tokenizer has chat template
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        print("=============================================================")
        print("WARNING: Model has no chat template. DPOTrainer requires one.")
        print("            Adding a dummy chat template for now.            ")
        print("=============================================================")
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for compatibility # TODO: What's this for? and is this only to be done if no chat template already exists?

    # Create the DPO trainer
    model_dir = os.path.join(args.output_dir, args.model_path)
    dpo_kwargs = {
        "output_dir": model_dir,
    }
    if args.debug:
        dpo_kwargs |= {
            "max_steps": 1,
            "logging_steps": 1,
            "per_device_train_batch_size": 1,
        }

    training_args = DPOConfig(**dpo_kwargs)
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=test_split if test_split else None,
        processing_class=tokenizer,
    )

    trainer.train()

    # Export command line args for reproducibility
    with open(os.path.join(model_dir, "args.json"), "w") as f_out:
        json.dump(vars(args), f_out)