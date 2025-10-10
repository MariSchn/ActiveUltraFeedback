from accelerate import Accelerator
from accelerate.utils import gather_object
from collections import deque
from datasets import Dataset, load_from_disk
import json
import os
import time
import torch
from torch.utils.data import DataLoader

from activeuf.acquisition_function import init_acquisition_function
from activeuf.new_loop.arguments import get_args
from activeuf.new_loop.utils import init_reward_model_tokenizer_trainer, custom_collate, custom_decollate
from activeuf.oracle.oracles import init_oracle
from activeuf.utils import get_logger, set_seed

# RUN
# python -m activeuf.new_loop.run --config_path activeuf/new_loop/run.yaml

def compute_rewards(samples, reward_model = None) -> torch.tensor:
    n_samples = len(samples)
    n_completions_per_sample = len(samples[0]["completions"])
 
    # TODO: make acquisition function type an attr of the acquisition function
    if reward_model is None:
        return torch.zeros(
            (n_samples, n_completions_per_sample, 3),
            dtype=torch.float32,
        )

    def get_features_yielder():
        for sample in samples:
            for completion in sample["completions"]:
                yield torch.tensor(completion["features"])

    features_yielder = get_features_yielder()
    rewards_batch = []
    while True:
        features_mbatch = []
        for _ in range(args.compute_reward_batch_size):
            try:
                features_mbatch.append(next(features_yielder))
            except StopIteration:
                break
        if not features_mbatch:
            break
        features_mbatch = torch.stack(features_mbatch).to(reward_model.device)

        with torch.no_grad():
            output = reward_model(features=features_mbatch)

        rewards_batch.extend(output["rewards"].cpu())

    torch.cuda.empty_cache()
    rewards_batch = torch.stack(rewards_batch).view(n_samples, -1, 3)

    return rewards_batch

def get_acquired(samples, acquired_idxs, tokenizer = None):
    acquired = []
    for sample, (a, b) in zip(samples, acquired_idxs):
        completions = sample["completions"]

        acquired.append({
            "prompt_id": sample["prompt_id"],
            "prompt": sample["prompt"],
            "source": sample["source"],
            "row_id": sample["row_id"],

            "response_text_1": completions[a]["response_text"],
            "1_model": completions[a]["model"],
            "1_score": completions[a]["overall_score"],

            "response_text_2": completions[b]["response_text"],
            "2_model": completions[b]["model"],
            "2_score": completions[b]["overall_score"],
        })

    if tokenizer is None:
        return acquired

    messages = []
    for sample, (a, b) in zip(samples, acquired_idxs):
        for idx in [a, b]:
            messages.append([
                {"role": "user", "content": sample["prompt"]},
                {
                    "role": "assistant", 
                    "content": sample["completions"][idx]["response_text"],
                },
            ])
    inputs = tokenizer(
        tokenizer.apply_chat_template(messages, tokenize=False),
        padding="longest",
        max_length=args.max_length,
        truncation=True,
        return_tensors="pt",
    )
    dim = inputs["input_ids"].shape[-1]
    input_ids = inputs["input_ids"].view(-1, 2, dim)
    attention_mask = inputs["attention_mask"].view(-1, 2, dim)

    for i, (a,b) in enumerate(acquired_idxs):
        acquired[i].update({
            "features_1": samples[i]["completions"][a]["features"],
            "input_ids_1": input_ids[i][0].tolist(),
            "attention_mask_1": attention_mask[i][0].tolist(),

            "features_2": samples[i]["completions"][b]["features"],
            "input_ids_2": input_ids[i][1].tolist(),
            "attention_mask_2": attention_mask[i][1].tolist(),
        })
    return acquired

if __name__ == "__main__":
    args = get_args()
    logger = get_logger(__name__, args.logs_path)

    # export args
    accelerator = Accelerator()
    n_processes = accelerator.num_processes
    if accelerator.is_main_process:
        with open(args.args_path, "w") as f_out:
            json.dump(vars(args), f_out)
        
    # setup wandb
    if args.report_to == "wandb" and accelerator.is_main_process:
        os.environ.setdefault("WANDB_DIR", args.wandb_dir)
    
    # GPU cleanup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if args.seed:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    logger.info(f"Preparing acquisition function ({args.acquisition_function})")
    acquisition_function = init_acquisition_function(
        args.acquisition_function,
        **args.acquisition_function_config.get(args.acquisition_function, {})
    )

    logger.info(f"Preparing oracle ({args.oracle_name})")
    oracle = init_oracle(args.oracle_name)

    # TODO but low priority: use a init function to allow for different reward models
    # TODO bypass pipeline and construct directly?
    # TODO: ask whether wandb callback for kpi logging is still necessary, given how it's set to False by default
    logger.info(f"Preparing reward model, tokenizer, and trainer ({args.reward_model})")
    reward_trainer_config = args.reward_trainer_config[args.reward_model]
    reward_model, tokenizer, trainer = init_reward_model_tokenizer_trainer(
        args,
        batch_size=reward_trainer_config["effective_batch_size"] // n_processes,
    )

    # wait for everyone to finish loading the models
    accelerator.wait_for_everyone()

    logger.info(f"Loading prompts from {args.inputs_path}")
    dataset = load_from_disk(args.inputs_path)
    if args.debug:
        dataset = dataset.select(range(100))
    if "row_id" not in dataset.column_names:
        dataset = dataset.add_column("row_id", list(range(len(dataset))))
    logger.info(f"# Prompts: {len(dataset)}")
    
    # TODO: find a nicer place for this someday. for now assume that everything has precomputed features
    if "features" not in dataset[0]["completions"][0]:
        logger.info("Please precompute features before running this script")
        exit(1)

    # TODO: figure out how to continue from a previous run of the loop
    # TODO: generalise replay buffer part because not all reward models will need it
    output = []
    replay_buffer = deque(maxlen=reward_trainer_config["replay_buffer_size"])
    dataset = dataset.shuffle(seed=args.seed)
    n_batches = len(dataset) // args.outer_loop_batch_size

    logger.info(f"Starting dataset generation loop")
    for batch_idx in range(n_batches):
        if accelerator.is_main_process:
            logger.info(f"Step {batch_idx+1} / {n_batches}")

        subset = dataset.select(range(
            batch_idx * args.outer_loop_batch_size, 
            min( (batch_idx+1) * args.outer_loop_batch_size, len(dataset))
        ))
        dataloader = DataLoader(
            subset,
            batch_size=len(subset) // n_processes,
            collate_fn=custom_collate,
            shuffle=False,
        )
        dataloader = accelerator.prepare(dataloader)

        annotated_batch = []
        for collated_minibatch in dataloader:
            samples_local = custom_decollate(collated_minibatch)

            start = time.time()
            reward_model.eval()
            rewards_local = compute_rewards(samples_local, reward_model)
            logger.info(f"- Reward computation took {time.time() - start:.2f}s")

            start = time.time()
            acquired_idxs_local = torch.tensor(
                acquisition_function(*rewards_local.unbind(-1))
            )
            logger.info(f"- Acquisition function took {time.time() - start:.2f}s")

            start = time.time()
            if args.acquisition_function == "random":
                acquired_local = get_acquired(samples_local, acquired_idxs_local)
            else:
                acquired_local = get_acquired(
                    samples_local, acquired_idxs_local, tokenizer)
            logger.info(f"- Preparing acquired batch took {time.time() - start:.2f}s")

            start = time.time()
            annotated_local = oracle(acquired_local)
            logger.info(f"- Oracle annotation took {time.time() - start:.2f}s")

            # TODO: wandb KPI reporting

            start = time.time()
            accelerator.wait_for_everyone()
            annotated_minibatch = gather_object(annotated_local)
            print(f"- Gathering annotations took {time.time() - start:.2f}s")


            print(annotated_minibatch[0].keys())
            annotated_batch += annotated_minibatch
        
    
    if accelerator.is_main_process and len(output) > 0:
        logger.info(f"Saving generated dataset to {args.output_path}")
        Dataset.from_list(output).save_to_disk(args.output_path)