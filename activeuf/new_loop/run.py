from accelerate import Accelerator
from accelerate.utils import gather_object
from collections import defaultdict, deque
from datasets import Dataset, load_from_disk
import json
import os
import random
import time
import torch
from torch.utils.data import DataLoader
import wandb

from activeuf.acquisition_function import init_acquisition_function
from activeuf.new_loop.arguments import get_args
from activeuf.new_loop import utils as loop_utils
from activeuf.oracle.oracles import init_oracle
from activeuf.utils import get_logger, set_seed

# RUN
# python -m activeuf.new_loop.run --config_path activeuf/new_loop/run.yaml

if __name__ == "__main__":
    # prepare args and logger
    args = get_args()
    logger = get_logger(__name__, args.logs_path)
    with open(args.args_path, "w") as f_out:
        json.dump(vars(args), f_out)
        
    # env setup
    accelerator = Accelerator()
    n_processes = accelerator.num_processes

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

    logger.info(f"Preparing reward model, tokenizer, and trainer ({args.reward_model})")
    reward_model, trainer, trainsize, initial_regularization = loop_utils.init_reward_model_trainer(
        args, n_processes
    )
    if accelerator.is_main_process and args.report_to == "wandb":
        os.environ.setdefault("WANDB_DIR", args.wandb_dir)
        trainer.add_callback(loop_utils.WandbStepLoggerCallback())

    # wait for everyone to finish loading the models
    accelerator.wait_for_everyone()

    logger.info(f"Loading prompts from {args.inputs_path}")
    dataset = load_from_disk(args.inputs_path)
    if args.debug:
        dataset = dataset.select(range(200))
    if "row_id" not in dataset.column_names:
        dataset = dataset.add_column("row_id", list(range(len(dataset))))    
    if "features" not in dataset[0]["completions"][0]:
        logger.info("Please precompute features before running this script")
        exit(1)
    dataset = dataset.shuffle(seed=args.seed)
    logger.info(f"# Prompts: {len(dataset)}")

    # prepare output container and trainer replay buffer
    output = []
    replay_buffer = deque(maxlen=args.replay_buffer_size)

    logger.info(f"Starting dataset generation loop")
    outer_dataloader = DataLoader(
        dataset, 
        batch_size=args.outer_loop_batch_size,
        collate_fn=lambda x: x,
        shuffle=False,
        drop_last=False,
    )
    for outer_batch_idx, outer_batch in enumerate(outer_dataloader):
        reward_model.eval()
        if accelerator.is_main_process:
            logger.info(f"Step {outer_batch_idx+1} / {len(outer_dataloader)}")

        dataloader = DataLoader(
            outer_batch,
            batch_size=max(1, len(outer_batch) // n_processes),
            collate_fn=loop_utils.custom_collate,
            shuffle=False,
            drop_last=False,
        )
        logger.info(f"- # Minibatches: {len(dataloader)}")
        dataloader = accelerator.prepare(dataloader)

        annotated_batch = []
        kpi_samplewise_batch = []
        kpi_batch = defaultdict(list)
        for collated_minibatch in dataloader:
            samples_local = loop_utils.custom_decollate(collated_minibatch)

            start = time.time()
            rewards_local = loop_utils.compute_rewards(
                samples_local, reward_model, args.compute_reward_batch_size)
            logger.info(f"- Reward computation took {time.time() - start:.2f}s")

            start = time.time()
            acquired_idxs_local = torch.tensor(
                acquisition_function(*rewards_local.unbind(-1))
            )
            logger.info(f"- Acquisition function took {time.time() - start:.2f}s")

            start = time.time()
            acquired_local = loop_utils.get_acquired(
                samples_local, acquired_idxs_local)
            logger.info(f"- Preparing acquired batch took {time.time() - start:.2f}s")

            start = time.time()
            annotated_local = [
                loop_utils.restructure_sample(x) for x in oracle(acquired_local)
            ]
            logger.info(f"- Oracle annotation took {time.time() - start:.2f}s")

            start = time.time()
            kpi_samplewise_local, kpi_minibatchwise_local = loop_utils.compute_kpi(
                rewards_local, acquired_idxs_local,
            )
            logger.info(f"- KPI computation took {time.time() - start:.2f}s")

            start = time.time()
            accelerator.wait_for_everyone()
            annotated_batch += gather_object(annotated_local)
            kpi_samplewise_batch += gather_object(kpi_samplewise_local)
            for key, val in gather_object(kpi_minibatchwise_local).items():
                kpi_batch[key].append(val)

            logger.info(f"- Gathering data from processes took {time.time() - start:.2f}s")

        if accelerator.is_main_process:
            # remove features from being saved to disk
            output += [
                {key: val for key, val in x.items() if "features" not in key} 
                for x in annotated_batch
            ]
            logger.info(f"Current output dataset size: {len(output)}")

            if outer_batch_idx % args.save_every_n_outer_batches == 0:
                logger.info(f"Writing output dataset to {args.output_path}")
                Dataset.from_list(output).save_to_disk(args.output_path)

            if args.report_to == "wandb":
                logger.info("Reporting KPIs to WandB")
                wandb.init(
                    project=args.wandb_project,
                    id=args.kpi_run_id,
                    config=vars(args),
                    resume=(outer_batch_idx > 0),
                    allow_val_change=True,
                )
                for idx, kpi in enumerate(kpi_samplewise_batch):
                    if idx == len(kpi_samplewise_batch) - 1:
                        # log batch-level metrics alongside last sample
                        for key, val in kpi_batch.items():
                            kpi[key] = sum(val) / len(val)
                    wandb.log(kpi)

                # set trainer wandb step to be the same as the step used for the most recent log
                loop_utils.TRAINER_WANDB_STEP = wandb.run.step - 1
                wandb.finish()

        if trainer is None:
            logger.info("Skipping reward model training")
            continue

        if accelerator.is_main_process and args.report_to == "wandb":
            wandb.init(
                project=args.wandb_project,
                id=args.trainer_run_id,
                config=vars(args),
                resume=outer_batch_idx > 0,
                allow_val_change=True,
            )
        
        logger.info(f"Adding fresh batch to replay buffer, then subsampling {trainsize} for training")
        # features are precomputed, so input_ids and attention_mask are not needed and we can just feed a dummy tensor to make trainer happy
        dummy_tensor = torch.zeros((1,), dtype=torch.long).to(reward_model.device)
        for idx, x in enumerate(annotated_batch):
            replay_buffer.append({
                "input_ids_chosen": dummy_tensor,
                "attention_mask_chosen": dummy_tensor,
                "input_ids_rejected": dummy_tensor,
                "attention_mask_rejected": dummy_tensor,
                "features_chosen": x["features_chosen"],
                "features_rejected": x["features_rejected"],
            })
        trainer.train_dataset = Dataset.from_list(random.sample(
            replay_buffer, min(len(replay_buffer), trainsize),
        ))

        logger.info(f"Updating regularization")
        trainer.args.regularization_towards_initial_weights = loop_utils.update_regularization(
            initial_regularization, 
            outer_batch_idx, 
            args.outer_loop_batch_size,
            args.reward_trainer_config[args.reward_model],
        )

        reward_model.train()
        start = time.time()
        trainer.train()
        logger.info(f"Reward model training took {time.time() - start:.2f}s")

        # cleanup
        torch.cuda.empty_cache()
        if accelerator.is_main_process and args.report_to == "wandb":
            wandb.finish()

    if accelerator.is_main_process and len(output) > 0:
        logger.info(f"Saving generated dataset to {args.output_path}")
        Dataset.from_list(output).save_to_disk(args.output_path)