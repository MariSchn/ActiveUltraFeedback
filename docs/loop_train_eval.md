# Loop Train Eval Pipeline

Run the Active Learning Loop, DPO + RM Training and DPO + RM Training Evals in one script to make experiments faster and enable easy sweeping

## Overview

The `loop_train_eval.sbatch` script runs the complete active learning pipeline in a single SLURM job:

1. **Active Learning Loop** - Runs the Active Learning Loop to generate a dataset
2. **DPO + RM Training** - Trains a DPO model and a Reward Model on the generated dataset (runs in parallel on separate nodes)
3. **DPO + RM Evaluation** - Evaluates the DPO model on benchmarks (GSM8K, Minerva Math, IFEval, TruthfulQA) and the RM on RewardBench (in parallel on separate nodes)
4. **WandB Update** - Aggregates all metrics and updates the loop WandB run 

### Example Run Command

```bash
sbatch scripts/loop_train_eval.sbatch
```

## Output Structure

After a successful run, outputs are organized as follows:

```
${SCRATCH}/models/
├── <BASE_RM_OUTPUT_DIR>/
│   └── <run_id>/
│       ├── config.json
│       ├── model.safetensors
│       ├── training.log
│       └── results/
│           ├── gsm8k_tulu/
│           ├── minerva_math_tulu/
│           ├── ifeval_tulu/
│           └── truthfulqa_tulu/
└── <BASE_DPO_OUTPUT_DIR>/
    └── <run_id>/
        ├── config.json
        ├── model.safetensors
        ├── training.log
        └── metrics.json
```

## Crashed Runs

If the pipeline fails partway through, use these scripts to run missing runs without re-running everything.

### `run_missing_trainings.sh`


This script assumes that the Active Learninng Loop finished and the model trainings failed. If the Active Learning Loop failed, re-run the entire script. It ...

- Scans the dataset directory for all generated datasets
- Checks if each dataset has a corresponding RM model (looking for for `config.json`)
- Checks if each dataset has a corresponding DPO model (looking for `config.json`)
- Submits SLURM jobs for any missing trainings

The script takes the following args:

- `loop_base_dir`: Directory containing datasets. The script will try to start trainings for all subdirs of this directory
- `rm_base_dir`: Output directory for the reward models. New models will be saved using the datasets name. I.e. if there is a datset at `<loop_base_dir>/<dataset_name>` the output reward model will be written to `<rm_base_dir>/<dataset_name>`
- `dpo_base_dir`: Output directory for the DPO models. 


```bash
./scripts/run_missing_trainings.sh \
    --loop_base_dir /path/to/datasets \
    --rm_base_dir /path/to/reward_models \
    --dpo_base_dir /path/to/dpo_models
```

### `run_missing_evals.sh`

This script assumes that model trainings finished but evaluations failed. It:

- Scans the RM model directory and checks for `metrics.json` (RewardBench results)
- Scans the DPO model directory and checks for all benchmark results:
  - `results/gsm8k_tulu/metrics.json`
  - `results/ifeval_tulu/metrics.json`
  - `results/minerva_math_tulu/metrics.json`
  - `results/truthfulqa_tulu/metrics.json`
- Submits individual SLURM jobs for each missing evaluation
- Updates WandB with results after each evaluation completes

The script takes the following args:

- `rm_base_dir`: Directory containing trained reward models. The script will check each subdir for missing `metrics.json`
- `dpo_base_dir`: Directory containing trained DPO models. The script will check each subdir for missing benchmark results in `results/`

```bash
./scripts/run_missing_evals.sh \
    --rm_base_dir /path/to/reward_models \
    --dpo_base_dir /path/to/dpo_models
```

### `run_missing_trainings_and_evals.sh`

This script combines both training and evaluation recovery. Use this when both trainings and evaluations may have failed. It:

- Scans the dataset directory for all generated datasets
- Checks if each dataset has a corresponding RM model (looking for `config.json`)
- Checks if each dataset has a corresponding DPO model (looking for `config.json`)
- Submits SLURM jobs for any missing trainings
- Runs evaluations after each training completes

The script takes the following args:

- `dataset_base_dir`: Directory containing datasets. The script will try to start trainings for all subdirs of this directory
- `rm_base_dir`: Output directory for the reward models. New models will be saved using the dataset name. I.e. if there is a dataset at `<dataset_base_dir>/<dataset_name>` the output reward model will be written to `<rm_base_dir>/<dataset_name>`
- `dpo_base_dir`: Output directory for the DPO models

```bash
./scripts/run_missing_trainings_and_evals.sh \
    --dataset_base_dir /path/to/datasets \
    --rm_base_dir /path/to/reward_models \
    --dpo_base_dir /path/to/dpo_models
```

## Sweeping

The script supports WandB sweeps for running hyperparameter searches.

1. Create a sweep config file (see `configs/sweeps/full_sweep.yaml` for an example):

```yaml
program: scripts/loop_train_eval.sbatch
method: grid
metric:
  name: "Rewardbench/Mean"
  goal: maximize

command:
  - scripts/loop_train_eval.sbatch 
  - ${args}

parameters:
  outer_loop_batch_size:
    values: [64, 256, 1024]
  replay_buffer_factor:
    values: [10, 100, 1000]
  enn.regularization.initial_value:
    values: [1.0, 10.0, 100.0]
```

2. Create the sweep on WandB (requires wandb CLI - install the uv venv as described in the README):

```bash
wandb sweep --entity ActiveUF --project loop configs/sweeps/full_sweep.yaml
```

3. Launch sweep agents using the sweep runner:

```bash
# Replace <SWEEP_ID> with the ID from step 2
# Replace <N> with the number of runs you want to launch
# Replace <MAX_PARALLEL> with the max number of jobs to run simultaneously
sbatch --array=1-<N>%<MAX_PARALLEL> ./scripts/sweep_runner.sbatch
```

Each job will pull hyperparameters from WandB and run the full pipeline with those settings.