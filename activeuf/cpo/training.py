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


def argument_override(args, training_config, lora_config):
    if args.learning_rate is not None:
        training_config["learning_rate"] = args.learning_rate
    if args.beta is not None:
        training_config["beta"] = args.beta
    if args.simpo_gamma is not None:
        training_config["simpo_gamma"] = args.simpo_gamma
    if args.per_device_train_batch_size is not None:
        training_config["per_device_train_batch_size"] = (
            args.per_device_train_batch_size
        )
    if args.gradient_accumulation_steps is not None:
        training_config["gradient_accumulation_steps"] = (
            args.gradient_accumulation_steps
        )
    if args.warmup_ratio is not None:
        training_config["warmup_ratio"] = args.warmup_ratio
    if args.loss_type is not None:
        training_config["loss_type"] = args.loss_type
    if args.lora_r is not None:
        lora_config["r"] = args.lora_r
    if args.lora_alpha is not None:
        lora_config["lora_alpha"] = args.lora_alpha
    if args.seed is not None:
        training_config["seed"] = args.seed


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
    argument_override(args, training_config, lora_config)
    if args.seed is not None:
        config["seed"] = args.seed

    peft_config = LoraConfig(**lora_config) if lora_config else None

    if process_id == 0:
        print("args:")
        print(args)
        print("Using config:")
        print(yaml.dump(config, default_flow_style=False))
        print("Using training config:")
        print(yaml.dump(training_config, default_flow_style=False))
        print("Using LoRA config:")
        print(yaml.dump(lora_config, default_flow_style=False))
        if peft_config:
            print("LoRA Config:")
            print(peft_config)
        else:
            print("Full parameter training (no LoRA).")

    # set seed for reproducibility
    if isinstance(config.get("seed"), int):
        set_seed(config.get("seed"))

    def sanitize_name(name):
        # Replace / and . with _
        return re.sub(r"[/.]", "-", name)

    run_name = f"{os.environ['SLURM_JOB_ID']}-{sanitize_name(os.path.basename(args.dataset_path.rstrip('/')))}"
    # adding HPs to run name (lr_rate, simpo_gamma, beta)
    run_name += f"-{training_config['loss_type']}-lr{training_config['learning_rate']}-sg{training_config['simpo_gamma']}-b{training_config['beta']}-seed{config.get('seed', 'NA')}"
    # if using LoRA, add lora_r and lora_alpha to run name
    if peft_config is not None:
        run_name += f"-loraR{peft_config.r}-loraA{peft_config.lora_alpha}"
    else:
        run_name += "-full"
    print(f"Run name: {run_name}")

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

    if process_id == 0:
        print("Training arguments:")
        print(training_args)

    # --- 2. Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with accelerator.main_process_first():
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
            dataset = dataset.select(range(64))

        print(f"Original dataset size: {len(dataset)}")

        # Necessary fix for now...
        def filter_rows(example):
            example = maybe_extract_prompt(example)
            example = maybe_apply_chat_template(example, tokenizer=tokenizer)
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]

            def build_tokenized_answer(prompt, answer):
                full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
                prompt_input_ids = tokenizer(prompt, add_special_tokens=False)[
                    "input_ids"
                ]
                answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
                answer_attention_mask = full_tokenized["attention_mask"][
                    len(prompt_input_ids) :
                ]
                full_concat_input_ids = np.concatenate(
                    [prompt_input_ids, answer_input_ids]
                )
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
    print("Shuffling dataset...")
    dataset = dataset.shuffle(seed=config.get("seed"))

    # --- 4. Model & PEFT ---
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )

    trainer = CPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if training_args.process_index == 0 and peft_config is not None:
        trainer.model.print_trainable_parameters()

    if training_args.process_index == 0:
        print("Starting SimPO training...")

    trainer.train()

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
        # Save the config for reproducibility
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        with open(os.path.join(output_dir, "config_used.yaml"), "w") as f_out:
            yaml.dump(config, f_out, default_flow_style=False)

    if peft_config is not None:
        if trainer.is_world_process_zero():
            print("LoRA detected: Unwrapping and merging adapters...")

        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)

        unwrapped_model = unwrapped_model.merge_and_unload()

        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=trainer.is_world_process_zero(),
            safe_serialization=True,
        )
    else:
        if trainer.is_world_process_zero():
            print("Full Finetune: Saving model (Collective Operation)...")
        trainer.save_model(output_dir)

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
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument("--beta", type=float, default=None, help="Override beta value")
    parser.add_argument(
        "--simpo_gamma", type=float, default=None, help="Override simpo gamma"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=None, help="Override warmup ratio"
    )
    parser.add_argument("--lora_r", type=int, default=None, help="Override LoRA rank")
    parser.add_argument(
        "--lora_alpha", type=int, default=None, help="Override LoRA alpha"
    )
    parser.add_argument(
        "--loss_type", type=str, default=None, help="Override loss type (cpo/simpo)"
    )

    parser.add_argument("--seed", type=int, default=None, help="Override random seed")

    args = parser.parse_args()
    main(args)
