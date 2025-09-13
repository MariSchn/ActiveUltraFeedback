import argparse
import os
import yaml

# from trl import DPOConfig
from trl.data_utils import apply_chat_template, extract_prompt
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, load_dataset

from accelerate import Accelerator

from activeuf.configs import *
from activeuf.utils import *
from activeuf.dpo.trainer import NormedDPOConfig, NormedDPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        help="Path to the YAML training config.")
    parser.add_argument("--slurm_job_id", type=str,
                        help="SLURM Job ID associated with this run")
    return parser.parse_args()


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
                "Unknown dataset format. Expected 'train' or 'train_prefs' split.")
    else:
        train_dataset = dataset

    return train_dataset


if __name__ == "__main__":
    # load env vars, args, configs
    setup()
    args = parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config["slurm_job_id"] = args.slurm_job_id

    lora_config = config.get("lora", {})
    training_config = config.get("training", {})

    # prepare output dir based on SLURM job id and run name
    run_name = f"{args.slurm_job_id}-{config['base_run_name']}"
    output_dir = os.path.join(config["base_output_dir"], run_name)

    # set seed for reproducibility
    if isinstance(config.get("seed"), int):
        set_seed(config.get("seed"))

    # prepare accelerator and torch stats
    accelerator = Accelerator()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # send config file to wandb
    if accelerator.is_main_process:
        wandb.init(name=run_name)
        wandb.config.update(config)
        artifact = wandb.Artifact(run_name, type="config")
        artifact.add_file(args.config_path)
        wandb.log_artifact(artifact)
    
    # let all ranks wait here so that W&B is ready before training starts
    accelerator.wait_for_everyone()

    if config.get("torch_dtype", "bfloat16") == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # load dataset, remove problematic columns
    print(f"Loading dataset from {config["train_path"]}...\n")
    dataset_path = config["train_path"]
    # Load dataset
    try:
        dataset = load_dataset(dataset_path)
    except Exception as e:
        try:
            dataset = load_from_disk(dataset_path)
        except Exception as e:
            print(f"Failed to load remote or local datasets: {e}")
            exit(-1)

    dataset = process_dataset(dataset)

    for column in ["messages", "prompt"]:
        try:
            dataset = dataset.remove_columns(column)
        except:
            print(f"Unable to remove {column=} from dataset")

    # limit dataset if in debug mode
    if config.get("debug"):
        dataset = dataset.select(range(100))

    # load tokenizer, then use it to remove overly long samples
    model_path = config["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # remove samples where prompt+chosen or prompt+rejected exceeds max length
    if training_config["max_length"]:
        temp = dataset.map(extract_prompt)
        temp = temp.map(apply_chat_template, fn_kwargs={
                        "tokenizer": tokenizer})
        temp = temp.map(lambda _: NormedDPOTrainer.tokenize_row(
            _,
            processing_class=tokenizer,
            max_prompt_length=None,
            max_completion_length=None,
            add_special_tokens=True,
        ))

        old_n = len(dataset)
        print(f"Original number of samples: {old_n}")
        def check_if_short(x: dict) -> dict[str, bool]:
            return {
                "is_short": len(x["prompt_input_ids"]) + len(x["chosen_input_ids"]) <= training_config["max_length"] or \
                len(x["prompt_input_ids"]) + len(x["rejected_input_ids"]) <= training_config["max_length"]
            }
        temp = temp.map(check_if_short)
        idxs = [i for i, _ in enumerate(temp["is_short"]) if _]
        dataset = dataset.select(idxs)
        print(f"Number of samples removed due to length constraints: {old_n - len(dataset)}")

    dataset = dataset.select_columns(["chosen", "rejected"])

    # create lora version of model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch_dtype)
    peft_config = LoraConfig(**lora_config)
    try:
        model = get_peft_model(model, peft_config)
    except Exception as e:
        print(f"Failed to apply PEFT model: {e}")
        print("Falling back to manually identifying target modules\n")

        target_modules = []
        if "llama" in model_path.lower():
            for name, _ in model.named_modules():
                if any(_ in name for _ in ["query", "key", "value"]):
                    target_modules.append(name)
            peft_config.target_modules = target_modules
        else:
            print(
                f"Target module patterns for {model_path} must be implemented manually")
            raise

    # create DPO trainer
    trainer_config = NormedDPOConfig(
        run_name=run_name, output_dir=output_dir, dataset_num_proc=accelerator.num_processes, **training_config)
    trainer = NormedDPOTrainer(
        model=model,
        args=trainer_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    print(trainer.normalize_logps)
    trainer.train()

    # Save final model
    if trainer.is_world_process_zero():
        model = model.merge_and_unload()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")

    # Export config file for reproducibility
    out_path = os.path.join(output_dir, os.path.basename(args.config_path))
    with open(out_path, "w") as f_out:
        yaml.dump(config, f_out, default_flow_style=False)

if accelerator.is_main_process and wandb.run is not None:
    wandb.finish()