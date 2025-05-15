<!-- <div align="center"> </div> -->
# Active UltraFeedback

Active UltraFeedback is a scalable pipeline that generates a preference dataset to be used for alignment of large language models, only requiring a set of prompts as input. 

It improves the efficiency and quality of the dataset by utilizing uncertainty quantification and active learning to determine the most informative samples to be annotated/labeled, by an oracle (usually another LLM, but possibly a human). This drastically reduces the number of samples that need to be annotated, and subsequently the cost of creating the dataset, while still maintaining a high quality dataset.

The dataset is generated through a loop which takes a batch of inputs and performs the following to generate a batch of preference data samples:

1. **Completion Generation**: Generate a completion for each prompt using a large pool of LLMs.
2. **Uncertainty Reward Prediction**: Predict the reward of each completion using a reward model. Additionally, the reward model also outputs the uncertainty of its prediction.
3. **Acquisition Function**: An acquisition function is used to create pairs of two completions for every prompt, that maximize the information.
4. **Oracle Annotation**: The pairs of completions are then annotated by an oracle (usually another LLM, but possibly a human). The oracle is asked to choose the better completion of the two.
5. **Reward Model Training**: The reward model is then trained on the new preference data samples, and the process is repeated.

All of this is taken care of in the `TODO` script, which is the main script for generating the dataset. It takes care of the entire pipeline, from generating the completions to generating the preference data samples.

## Getting Started

### Quickstart

To get started with Active UltraFeedback, you can simply install the package using pip:

```bash
pip install -e .
```

Then, you can run the `TODO` script to generate the dataset. The script takes care of the entire pipeline, from generating the completions to generating the preference data samples.

```bash
TODO
```

### 1. Environment Setup

To setup the environment, you can use either Conda (using the `environment.yml`) or Docker (using the `Dockerfile`). The code was developed and tested using Docker and Podman, so we recommend setting up the environment using Docker.

#### Docker + Podman

To seutp the environment you first need to build the Docker image and then convert it to a Podman image. You can do this by running the following commands:

```bash
podman build -t activeuf:25.02-py3 .
enroot import -o ./activeuf.sqsh podman://activeuf:25.02-py3
```

Afterwards, you need to setup a `.toml` file that uses the Docker image you just built. You can do this by creating a file called `activeuf.toml` in the directory where you store your `.toml` files, e.g. `~/.edf/`. The file should look like this:

```toml
image = <path-to-your-image>

mounts = [
    <path-to-your-local-dir>:<path-to-your-container-dir>
]

workdir = <initial-dir-in-container>

[annotations]  # Optional annotations. For example:
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"
```

Afterwards, you can specify this file when running a container with SLURM using the `--environment <name-of-your-toml-file>` flag. For example:

```bash
srun --environment activeuf --pty bash
```

#### Conda
```bash
conda env create -f environment.yml
```

### 2. Installing our Package

## Dataset Generation

### Dataset Format
### Dataset Example

## Dataset Evaluation

