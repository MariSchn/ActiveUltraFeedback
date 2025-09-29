from accelerate import Accelerator
from collections import deque
from datasets import Dataset, DataLoader, load_from_disk
import json
import os
import time
import torch

from activeuf.acquisition_function import init_acquisition_function
from activeuf.active_learning_loop.arguments import get_args
from activeuf.active_learning_loop.utils import init_model_tokenizer_trainer, custom_collate_fn
from activeuf.oracle.oracles import init_oracle
from activeuf.utils import get_logger, set_seed

# RUN
# python -m activeuf.active_learning_loop.run --config_path activeuf/active_learning_loop/run.yaml

def compute_features():
    pass


def get_preference_batch(dataloader, reward_model, acquisition_function):

    for batch in dataloader:
        n_samples_in_batch = len(batch["prompt_id"])
        n_completions_per_sample = len(batch[0]["completions"])
        has_precomputed_features = "features" in batch[0]["completions"][0]

        # tokenization phase

        # reward computation phase

        # acquisition phase

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
    model, tokenizer, trainer = init_model_tokenizer_trainer(
        args,
        batch_size=reward_trainer_config["effective_batch_size"] // n_processes,
    )

    # wait for everyone to load the models
    accelerator.wait_for_everyone()

    logger.info(f"Loading prompts from {args.inputs_path}")
    dataset = load_from_disk(args.inputs_path)
    if args.debug:
        dataset = dataset.select(range(50))
    if "row_id" not in dataset.column_names:
        dataset = dataset.add_column("row_id", list(range(len(dataset))))
    logger.info(f"# Prompts: {len(dataset)}")
    
    # TODO: find a nicer place for this someday
    if "features" not in dataset[0]["completions"][0]:
        compute_features()

    # TODO: figure out how to continue from a previous run of the loop
    # TODO: make generalise replay buffer part because not all reward models will need it
    output = []
    replay_buffer = deque(maxlen=reward_trainer_config["replay_buffer_size"])
    dataset = dataset.shuffle(seed=args.seed)
    n_batches = len(dataset) // args.outer_loop_batch_size

    for i in range(n_batches):
        if accelerator.is_main_process:
            logger.info(f"Step {i} / {n_batches}")

        start = i * args.outer_loop_batch_size
        end = min( (i+1) * args.outer_loop_batch_size, len(dataset))
        batch_dataset = dataset.select(range(start, end))
        batch_dataloader = DataLoader(
            batch_dataset,
            batch_size=len(batch_dataset) // n_processes,
            collate_fn=custom_collate_fn,
            shuffle=False,
        )
        batch_dataloader = accelerator.prepare(batch_dataloader)

        preference_batch = get_preference_batch(
            batch_dataloader,
            acquisition_function,
        )
    
    if accelerator.is_main_process and len(output) > 0:
        logger.info(f"Saving generated dataset to {args.output_path}")
        Dataset.from_list(output).save_to_disk(args.output_path)