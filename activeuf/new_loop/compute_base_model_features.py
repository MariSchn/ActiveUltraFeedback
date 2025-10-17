from argparse import ArgumentParser
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset, load_from_disk
from functools import partial
import yaml

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rewarduq.models.reward_head_ensemble import RewardHeadEnsembleModel, RewardHeadEnsembleModelConfig

from activeuf.utils import set_seed, get_timestamp

# accelerate launch --config_file=activeuf/new_loop/accelerate.yaml -m activeuf.new_loop.compute_base_model_features --config_path activeuf/new_loop/compute_base_model_features.yaml

def collate_fn(batch: list[dict], tokenizer):
    """
    Collate function for dynamic padding per batch.
    Pads all sequences in the batch to the length of the longest sequence.
    """
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    temp_ids = [x["temp_id"] for x in batch]

    # Use tokenizer.pad to dynamically pad batch
    batch_inputs = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding="longest",           # pad to the longest sequence in this batch
        return_tensors="pt"
    )
    
    return temp_ids, batch_inputs

if __name__ == "__main__":
    accelerator = Accelerator()

    cli_parser = ArgumentParser()
    cli_parser.add_argument(
        "--config_path", required=True, help="Path to the YAML config")
    config_path = cli_parser.parse_args().config_path
    with open(config_path, "r") as f:
        args = yaml.safe_load(f)
    args["timestamp"] = get_timestamp(more_detailed=True)

    args_path = f"{args['inputs_path'].rstrip('/')}_features-{args['timestamp']}.args"
    if accelerator.is_main_process:
        with open(args_path, "w") as f_out:
            print(yaml.dump(args), file=f_out)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if args['seed']:
        set_seed(args['seed'])

    model = RewardHeadEnsembleModel(RewardHeadEnsembleModelConfig(
        **{k: v for k, v in args['enn']["model"].items() if not k.startswith("__")}))
    model.eval()

    dataset = load_from_disk(args['inputs_path'])
    if args['debug']:
        dataset = dataset.select(range(500))
    n_response_texts_per_prompt = len(dataset[0]["completions"])
        
    n = len(dataset)
    dataset = dataset.add_column("prompt_idx", list(range(n)))

    per_proc = (n + accelerator.num_processes - 1) // accelerator.num_processes
    start = accelerator.process_index * per_proc
    end = min(start + per_proc, n)
    _dataset = dataset.select(range(start, end))

    _flattened_inputs = []
    for x in tqdm(_dataset, disable=not accelerator.is_main_process):
        messages = [
            [
                {"role": "user", "content": x["prompt"]},
                {"role": "assistant", "content": completion["response_text"]},
            ] for completion in x["completions"]
        ]
        messages_str = model.tokenizer.apply_chat_template(messages, tokenize=False)

        inputs = model.tokenizer(
            messages_str,
            padding="do_not_pad",
            truncation=True,
            max_length=args['max_length'],
            return_tensors=None,
        )

        for completion_idx in range(n_response_texts_per_prompt):
            _flattened_inputs.append({
                "temp_id": (x["prompt_idx"], completion_idx),
                "input_ids": inputs["input_ids"][completion_idx],
                "attention_mask": inputs["attention_mask"][completion_idx],
        })
    _flattened_inputs.sort(key=lambda x: -len(x["input_ids"]))
    _flattened_inputs = Dataset.from_list(_flattened_inputs)

    dataloader = DataLoader(
        _flattened_inputs,
        batch_size=args['batch_size'],
        num_workers=0,
        drop_last=False,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=model.tokenizer)
    )
    model = accelerator.prepare(model)

    accelerator.wait_for_everyone()
    outputs = []
    for temp_ids, inputs in tqdm(dataloader, disable=not accelerator.is_main_process):
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = model(output_only_features=True, **inputs).cpu()
        outputs += list(zip(temp_ids, features))

    accelerator.wait_for_everyone()
    outputs = gather_object(outputs)
    outputs.sort(key=lambda x: x[0])

    features = torch.stack([x[1] for x in outputs], dim=0)
    features = features.view(n, n_response_texts_per_prompt, -1)
    if accelerator.is_main_process:
        features_path = f"{args['inputs_path'].rstrip('/')}-features-{args['timestamp']}.pt"
        torch.save(features, features_path)