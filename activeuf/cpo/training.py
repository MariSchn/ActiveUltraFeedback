import torch
import os
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import CPOTrainer, CPOConfig
from peft import LoraConfig, get_peft_model
from argparse import ArgumentParser
import yaml
import wandb
import re
import random
from dotenv import load_dotenv
import huggingface_hub
from accelerate import Accelerator
from trl.data_utils import maybe_extract_prompt, maybe_apply_chat_template


# Problem with version of vllm and transformers,
# As utils imports everything together, I had to copy paste the setup function here.
def setup(login_to_hf: bool = False, login_to_wandb: bool = False) -> None:
    # load env variables
    load_dotenv(".env")
    load_dotenv(".env.local")

    if login_to_hf:
        huggingface_hub.login(os.getenv("HF_TOKEN"))

    if login_to_wandb:
        wandb.login(key=os.getenv("WANDB_TOKEN"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # False
    torch.backends.cudnn.benchmark = True  # maybe TRUE
    os.environ["PYTHONHASHSEED"] = str(seed)


"""
Prerequisites:
pip install transformers trl peft

accelerate launch --num_processes=4 --config_file=$SCRATCH/ActiveUltraFeedback/configs/accelerate/deepspeed2.yaml ./activeuf/cpo/training.py

accelerate launch --num_processes=4 --config_file=$SCRATCH/ActiveUltraFeedback/configs/accelerate/deepspeed2.yaml -m activeuf.cpo.training --config_path=$SCRATCH/ActiveUltraFeedback/configs/cpo_training.yaml --dataset_path=allenai/ultrafeedback_binarized_cleaned --output_dir=$SCRATCH/models/cpo

pip install --upgrade trl
"""


def process_dataset(dataset):
    # Dataset processing (Determining splits, restructuring (chosen/rejected columns))
    if isinstance(dataset, dict):
        if "train_prefs" in dataset:
            train_dataset = dataset["train_prefs"]
        elif "train" in dataset:
            train_dataset = dataset["train"]
        else:
            # TODO: More general way of handling dataset splits (They should come as arguments, for example).
            raise Exception(
                "Unknown dataset format. Expected 'train' or 'train_prefs' split."
            )
    else:
        train_dataset = dataset

    return train_dataset


def main(args):
    setup()
    accelerator = Accelerator()
    process_id = accelerator.process_index

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    lora_config = config.get("lora", {})
    training_config = config.get("training", {})

    if config.get("torch_dtype", "bfloat16") == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # set seed for reproducibility
    if isinstance(config.get("seed"), int):
        set_seed(config.get("seed"))

    peft_config = LoraConfig(
        **lora_config,
    )

    def sanitize_name(name):
        # Replace / and . with _
        return re.sub(r"[/.]", "-", name)

    run_name = f"{os.environ['SLURM_JOB_ID']}-{sanitize_name(os.path.basename(args.dataset_path.rstrip('/')))}"
    output_dir = os.path.join(args.output_dir, run_name)

    # send config file to wandb
    if process_id == 0 and training_config.get("report_to") == "wandb":
        wandb.init(name=run_name, entity="ActiveUF", project="CPO")
        wandb.config.update(config)
        artifact = wandb.Artifact(run_name, type="config")
        artifact.add_file(args.config_path)
        wandb.log_artifact(artifact)

    training_args = CPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        **training_config,
        dataset_num_proc=os.cpu_count(),
    )

    # --- 2. Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Data ---
    try:
        dataset = load_dataset(args.dataset_path)
    except Exception:
        try:
            dataset = load_from_disk(args.dataset_path)
        except Exception as e:
            print(f"Failed to load remote or local datasets: {e}")
            exit(-1)
    dataset = process_dataset(dataset)
    dataset = dataset.select_columns(["chosen", "rejected"])
    if args.debug:
        print("Debug mode enabled: using a smaller subset of the dataset.")
        dataset = dataset.select(range(16))

    def filter_prompts(example):
        prompt_msgs = example["chosen"][:-1]

        prompt_ids = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True
        )

        whole_thing_chosen = tokenizer.apply_chat_template(
            example["chosen"], tokenize=True, add_generation_prompt=True
        )
        whole_thing_rejected = tokenizer.apply_chat_template(
            example["rejected"], tokenize=True, add_generation_prompt=True
        )
        return (
            len(prompt_ids) < training_config["max_prompt_length"]
            and len(whole_thing_chosen) < training_config["max_length"]
            and len(whole_thing_rejected) < training_config["max_length"]
        )

    print(f"Original dataset size: {len(dataset)}")

    # Run the filter
    dataset = dataset.filter(
        filter_prompts,
        num_proc=os.cpu_count(),
        desc=f"Filtering prompts > {training_config['max_prompt_length']}",
    )
    print(f"Filtered dataset size: {len(dataset)} examples.")

    # Necessary fix for now...
    def filter_rows(example):
        example = maybe_extract_prompt(example)
        example = maybe_apply_chat_template(example, tokenizer=tokenizer)
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        def build_tokenized_answer(prompt, answer):
            full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
            prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
            answer_attention_mask = full_tokenized["attention_mask"][
                len(prompt_input_ids) :
            ]
            full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
            full_input_ids = np.array(full_tokenized["input_ids"])
            if len(full_input_ids) != len(full_concat_input_ids):
                raise ValueError(
                    "Prompt input ids and answer input ids should have the same length."
                )
            response_token_ids_start_idx = len(prompt_input_ids)
            if (
                prompt_input_ids
                != full_tokenized["input_ids"][:response_token_ids_start_idx]
            ):
                response_token_ids_start_idx -= 1
            prompt_input_ids = full_tokenized["input_ids"][
                :response_token_ids_start_idx
            ]
            prompt_attention_mask = full_tokenized["attention_mask"][
                :response_token_ids_start_idx
            ]
            if len(prompt_input_ids) != len(prompt_attention_mask):
                raise ValueError(
                    "Prompt input ids and attention mask should have the same length."
                )
            answer_input_ids = full_tokenized["input_ids"][
                response_token_ids_start_idx:
            ]
            answer_attention_mask = full_tokenized["attention_mask"][
                response_token_ids_start_idx:
            ]
            return dict(
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                input_ids=answer_input_ids,
                attention_mask=answer_attention_mask,
            )

        chosen_tokens = build_tokenized_answer(prompt, chosen)
        rejected_tokens = build_tokenized_answer(prompt, rejected)
        if len(chosen_tokens["prompt_input_ids"]) != len(
            rejected_tokens["prompt_input_ids"]
        ):
            return False
        return True

    dataset = dataset.filter(filter_rows, num_proc=os.cpu_count())
    print(f"Final dataset size after further filtering: {len(dataset)} examples.")
    print(dataset)
    # --- 4. Model & PEFT ---
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )

    model = get_peft_model(model, peft_config)

    if training_args.process_index == 0:
        model.print_trainable_parameters()

    trainer = CPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    if training_args.process_index == 0:
        print("Starting SimPO training...")

    trainer.train()

    if trainer.is_world_process_zero():
        model = model.merge_and_unload()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        with open(os.path.join(output_dir, "config_used.yaml"), "w") as f_out:
            yaml.dump(config, f_out, default_flow_style=False)

    if (
        process_id == 0
        and training_config.get("report_to") == "wandb"
        and wandb.run is not None
    ):
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, runs in debug mode with smaller dataset",
    )

    args = parser.parse_args()
    main(args)
