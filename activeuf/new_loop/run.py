from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from collections import deque
from dataclasses import asdict
from datasets import Dataset, load_from_disk
import os
import random
import time
import torch
from torch.utils.data import DataLoader
import wandb

from activeuf.acquisition_function import init_acquisition_function
from activeuf.new_loop.arguments import get_loop_args
from activeuf.new_loop import utils as loop_utils
from activeuf.oracle.oracles import init_oracle
from activeuf.utils import get_logger, get_timestamp, set_seed, convert_dataclass_instance_to_yaml_str

# RUN
# python -m pip install -e resources/rewarduq
# accelerate launch --config_file=activeuf/new_loop/accelerate.yaml -m activeuf.new_loop.run --config_path activeuf/new_loop/run.yaml

if __name__ == "__main__":
    accelerator = Accelerator()
    n_processes = accelerator.num_processes

    # prepare (and export) args
    if accelerator.is_main_process:
        timestamp = get_timestamp(more_detailed=True)
    else:
        timestamp = ""
    timestamp = broadcast_object_list([timestamp])[0]
    args = get_loop_args(timestamp)
    try:
        acquisition_function_args = asdict(
            getattr(args.acquisition_function, args.acquisition_function_type)
        )
    except:
        acquisition_function_args = {}
    if hasattr(args, args.reward_model_type):
        reward_args = getattr(args, args.reward_model_type)
    else:
        reward_args = None
    if accelerator.is_main_process:
        with open(args.args_path, "w") as f_out:
            print(convert_dataclass_instance_to_yaml_str(args), file=f_out)

    # env setup
    logger = get_logger(__name__, args.logs_path, accelerator)
    logger.info = loop_utils.main_process_only(logger.info, accelerator)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if args.seed:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.environ.setdefault("WANDB_DIR", args.wandb_dir)
        wandb.init(
            project=args.wandb_project,
            id=args.run_id,
            config=vars(args),
        )

    logger.info(f"Preparing acquisition function ({args.acquisition_function_type})")
    acquisition_function = init_acquisition_function(
        args.acquisition_function_type, **acquisition_function_args
    )

    logger.info(f"Preparing oracle ({args.oracle_name})")
    oracle = init_oracle(args.oracle_name)

    logger.info(f"Loading prompts from {args.inputs_path}")
    dataset = load_from_disk(args.inputs_path)
    assert "features" in dataset.column_names, "Dataset must have precomputed features"
    if args.debug:
        dataset = dataset.select(range(1001))
    dataset = dataset.shuffle(seed=args.seed)
    logger.info(f"# Prompts: {len(dataset)}")

    logger.info(
        f"Preparing reward model, tokenizer, and trainer ({args.reward_model_type})"
    )
    model, trainer = loop_utils.init_model_trainer(
        args.reward_model_type, reward_args, n_processes
    )
    if accelerator.is_main_process and trainer is not None:
        trainer.add_callback(loop_utils.WandbStepLoggerCallback(accelerator))
    tokenizer = model.tokenizer

    # prepare output container and trainer replay buffer
    output = []
    expected_output_size = len(dataset)
    replay_buffer = deque(maxlen=args.outer_loop_batch_size * args.replay_buffer_factor)

    logger.info(f"Starting dataset generation loop")
    outer_dataloader = DataLoader(
        dataset,
        batch_size=args.outer_loop_batch_size,
        collate_fn=lambda x: x,
        shuffle=False,
        drop_last=False,
    )
    for outer_batch_idx, outer_batch in enumerate(outer_dataloader):
        if model is not None:
            model.eval()

        if accelerator.is_main_process:
            logger.info(f"Step {outer_batch_idx + 1} / {len(outer_dataloader)}")

        dataloader = DataLoader(
            outer_batch,
            batch_size=max(1, -(len(outer_batch) // -n_processes)),
            collate_fn=loop_utils.custom_collate,
            shuffle=False,
            drop_last=False,
        )
        logger.info(f"- # Minibatches: {len(dataloader)}")
        dataloader = accelerator.prepare(dataloader)

        annotated_batch = []
        kpis_batch = []
        for collated_minibatch in dataloader:
            samples_local = loop_utils.custom_decollate(collated_minibatch)

            start = time.time()
            rewards_local = loop_utils.compute_rewards(
                samples_local, model, reward_args.inference_batch_size
            )
            logger.info(f"- Reward computation took {time.time() - start:.2f}s")

            start = time.time()
            acquired_idxs_local = torch.tensor(
                acquisition_function(*rewards_local.unbind(-1))
            )
            logger.info(f"- Acquisition function took {time.time() - start:.2f}s")

            start = time.time()
            acquired_local = loop_utils.get_acquired(samples_local, acquired_idxs_local)
            logger.info(f"- Preparing acquired batch took {time.time() - start:.2f}s")

            start = time.time()
            annotated_local = [
                loop_utils.restructure_sample(x) for x in oracle(acquired_local)
            ]
            logger.info(f"- Oracle annotation took {time.time() - start:.2f}s")

            start = time.time()
            kpis_local = loop_utils.compute_kpis(
                rewards_local,
                acquired_idxs_local,
            )
            logger.info(f"- KPI computation took {time.time() - start:.2f}s")

            start = time.time()
            annotated_batch += accelerator.gather_for_metrics(annotated_local)
            kpis_batch += accelerator.gather_for_metrics(kpis_local)
            logger.info(
                f"- Gathering data from processes took {time.time() - start:.2f}s"
            )

        # put batch-level KPIs alongside KPIs for final microbatch
        for idx, kpi in enumerate(kpis_batch):
            if idx == len(kpis_batch) - 1:
                for key, val in kpi.copy().items():
                    key2 = key.replace("per_sample", "per_batch")
                    if not key2.startswith("mean_"):
                        key2 = f"mean_{key2}"
                    kpi[key2] = sum(kpi2[key] for kpi2 in kpis_batch) / len(
                        kpis_batch
                    )

        logger.info(f"- Number of samples annotated in this batch: {len(annotated_batch)}")
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

        if trainer is None:
            if accelerator.is_main_process:
                logger.info("Reporting KPIs to WandB")
                for kpis in kpis_batch:
                    wandb.log(kpis)

            logger.info("Skipping reward model training")
            continue
        else:
            loop_utils.WANDB_LOGS_CACHE += kpis_batch

        start = time.time()
        trainsize = reward_args.effective_batch_size * reward_args.max_steps
        logger.info(
            f"Adding fresh batch to replay buffer, then subsampling {trainsize} for training"
        )
        # features are precomputed, so input_ids and attention_mask are not needed and we can just feed a dummy tensor to make trainer happy
        for idx, x in enumerate(annotated_batch):
            replay_buffer.append({
                "input_ids_chosen": torch.tensor([0]),
                "attention_mask_chosen": torch.tensor([0]),
                "features_chosen": x["features_chosen"],
                "input_ids_rejected": torch.tensor([0]),
                "attention_mask_rejected": torch.tensor([0]),
                "features_rejected": x["features_rejected"],
            })

        trainer.train_dataset = Dataset.from_list(random.sample(
            replay_buffer, min(len(replay_buffer), trainsize),
        ))
        trainer.train_dataset.set_format(
            type="torch", columns=trainer.train_dataset.column_names
        )
        loop_utils.MAX_TRAINER_LOGS_CACHE_SIZE = len(trainer.get_train_dataloader())
        
        trainer.args.regularization_towards_initial_weights = (
            loop_utils.get_new_regularization(
                n_done=len(output),
                n_total=expected_output_size,
                **asdict(reward_args.regularization),
            )
        )

        start = time.time()
        model.train()
        trainer.train()
        logger.info(f"Reward model training took {time.time() - start:.2f}s")

        # cleanup
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        wandb.finish()

    if accelerator.is_main_process and len(output) > 0:
        logger.info(f"Writing output dataset to {args.output_path}")
        Dataset.from_list(output).save_to_disk(args.output_path)
