# Active UltraFeedback

**Active UltraFeedback** is a scalable pipeline for generating high-quality preference datasets to align large language models (LLMs), requiring only a set of prompts as input.

It leverages **uncertainty quantification** and **active learning** to identify and annotate the most informative samples, drastically reducing annotation costs while maintaining high data quality. Annotations are provided by an oracle (typically another LLM, but can also be a human).

## 🔁 Pipeline Overview

Given a batch of prompts, the following steps are executed:

1. **Response Generation**: For each prompt in the batch, call multiple LLMs to each generate a response to the prompt.
2. **Uncertainty-Aware Reward Prediction**: An uncertainty-aware reward model predicts the reward and associated uncertainty of the responses for each prompt.  <br />
**Note**: This reward model is initialzied randomly in the beginning. 
3. **Pair Selection (Acquisition Function)**: Select which two responses should get (preference) annotated based on the rewards and uncertainties, using an acquisition function, e.g. Double Thompson Sampling.
4. **Oracle Annotation**: Annotate which response in the selected pairs is preferred (e.g., via another LLM or human feedback).
5. **Reward Model Training**: Train the uncertainty-aware reward model on the new preference data, then repeat the loop.

## 🚀 Quickstart

### 1. Installation

Install the package in editable mode:

```bash
pip install -e .
```

### 2. Running the Pipeline

Run the main dataset generation script:

```bash
python path/to/main_script.py  
```

### 3. Configuration (Optional)
To modify the pipeline parameters and steps, edit the configuration files in the `config/` directory. A quick overview:

TODO


## 🛠 Environment Setup

You can use **Docker/Podman** (recommended) or **Conda** (for local development).

### Option 1: Docker/Podman (Recommended)

Build the container image:

```bash
podman build -t activeuf:latest .
```

### Option 2: `uv` (For Local Use)

Create a `uv` environment with all dependencies. To install uv, simply run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Afterwards running the following command will install and synchronize all dependencies:

```bash
uv sync --dev
source .venv/bin/activate
```

Note: 

## 👨‍💻 Development Setup

For contributors and developers working on this project:

### 0. Set Up Environment (see above)

### 1. Set Up Pre-commit Hooks

This project uses `ruff` for linting and formatting. Install the pre-commit hooks to automatically format and lint your code before each commit:

```bash
pre-commit install
```

The pre-commit hooks will automatically:
- Run `ruff check --fix` to lint and auto-fix issues
- Run `ruff format` to format code

### 2. Manual Linting and Formatting

You can also run these tools manually:

```bash
# Format code
ruff format

# Lint code
ruff check

# Lint and auto-fix issues
ruff check --fix
```


## 📄 License

TODO
