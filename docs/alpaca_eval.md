# Alpaca Eval

This document describes how to run Alpaca Eval evaluations for DPO-trained models and combine the results.

---

## Overview

Alpaca Eval is an automated evaluation framework that assesses model outputs using a reference model (typically GPT-4 or a strong open-source model). The evaluation compares model completions on the Alpaca Eval dataset and generates win rates and other metrics.

---

## Scripts

### **Key Files**
- **`scripts/dpo/run_alpaca_eval.sbatch`**: SLURM batch script for running Alpaca Eval on a single model.
- **`scripts/dpo/combine_alpaca_eval_leaderboards.sh`**: Script to combine multiple leaderboard CSVs into a single file.
- **`resources/alpaca_eval/`**: The alpaca_eval repository (cloned automatically if needed).

---

## Running Alpaca Eval on a Single Model

### **Purpose**
Evaluate a single DPO-trained model using Alpaca Eval with a vLLM-based annotator.

### **Prerequisites**
1. The model must exist in the `models/dpo/` directory (or the directory specified by `MODELS_DIR`)
2. The annotator model (default: Llama 3.3 70B Instruct) must be available in your HuggingFace cache
3. Sufficient GPU resources (4 GPUs recommended for the annotator)

### **Command**
To run Alpaca Eval, submit the job using SLURM with required environment variables:

```bash
sbatch --export=ALL,MODEL_NAME="1sgwtpjp/v9no9em2",RESULTS_DIR="results/alpaca_eval/dpo" scripts/dpo/run_alpaca_eval.sbatch
```

### **Required Parameters**

- **`MODEL_NAME`**: The model name/path (e.g., `"1sgwtpjp/v9no9em2"`). This should match a subdirectory in `models/dpo/` (or `MODELS_DIR`). The script will create a config file at `resources/alpaca_eval/src/alpaca_eval/models_configs/${MODEL_NAME}/configs.yaml`
- **`RESULTS_DIR`**: Base directory for evaluation results (e.g., `"results/alpaca_eval/dpo"`)

### **Optional Parameters**

You can override these defaults via environment variables:

- **`MODELS_DIR`**: Base directory containing models (default: `models/dpo`)
- **`ANNOTATOR_CONFIG_NAME`**: Annotator config name (default: `weighted_alpaca_eval_vllm_llama3_70b`)
- **`ANNOTATOR_MODEL_NAME`**: Annotator model name (default: `meta-llama/Llama-3.3-70B-Instruct`)
- **`ANNOTATOR_GPU_MEM_UTILIZATION`**: GPU memory utilization for annotator (default: `0.7`)
- **`ANNOTATOR_TENSOR_PARALLEL_SIZE`**: Tensor parallel size for annotator (default: `4`)
- **`HF_CACHE`**: HuggingFace cache directory (default: `${SCRATCH}/huggingface`)
- **`ANNOTATOR_PORT`**: Port for vLLM server (default: `25125`)
- **`ANNOTATOR_API_KEY`**: API key for vLLM server (default: `token-abc123`)

### **Example with Custom Options**

```bash
sbatch --export=ALL,MODEL_NAME="1sgwtpjp/v9no9em2",RESULTS_DIR="results/alpaca_eval/dpo",ANNOTATOR_GPU_MEM_UTILIZATION="0.8" scripts/dpo/run_alpaca_eval.sbatch
```

### **What the Script Does**

1. **Start Annotator**: Launches a vLLM server for the annotator model in the background
2. **Setup**: Clones alpaca_eval repository if needed and installs it in edit mode
3. **Create Config**: Creates a custom model config file at `resources/alpaca_eval/src/alpaca_eval/models_configs/${MODEL_NAME}/configs.yaml` with the model path set to `${MODELS_DIR}/${MODEL_NAME}`
4. **Wait for Server**: Waits up to 10 minutes for the vLLM server to be ready
5. **Run Evaluation**: Executes `alpaca_eval evaluate_from_model` with the created config
6. **Cleanup**: Stops the vLLM server after evaluation completes

### **Output Structure**

Results are saved to: `${RESULTS_DIR}/${MODEL_NAME}/`

Key output files:
- **`model_outputs.json`**: Generated completions from the model
- **`weighted_alpaca_eval_vllm_llama3_70b/leaderboard.csv`**: Evaluation results and metrics
- **`weighted_alpaca_eval_vllm_llama3_70b/annotations.json`**: Detailed annotations
- **`annotator_server.log`**: Logs from the vLLM annotator server

### **SLURM Resources**

The script requests:
- **Nodes**: 1
- **GPUs**: 4 (for tensor-parallel annotator)
- **CPUs**: 32
- **Time**: 15 minutes (configurable via `--time` in the script)
- **Partition**: normal
- **Account**: a-infra01-1

---

## Combining Multiple Leaderboards

### **Purpose**
Combine multiple leaderboard CSV files from different model evaluations into a single CSV for easy comparison.

### **Command**
Run the script directly (no SLURM needed):

```bash
./scripts/dpo/combine_alpaca_eval_leaderboards.sh
```

Or with custom options:

```bash
./scripts/dpo/combine_alpaca_eval_leaderboards.sh --base-dir results/alpaca_eval/dpo --output results/alpaca_eval/combined_leaderboard.csv
```

### **Parameters**

- **`--base-dir`**: Base directory to search for CSV files (default: `results/alpaca_eval/dpo`)
- **`--output`**: Output file path (default: `results/alpaca_eval/combined_leaderboard.csv`)
- **`-h, --help`**: Show help message

### **How It Works**

The script:
1. Searches for CSV files matching the pattern: `BASE_DIR/*/*/weighted_alpaca_eval_vllm_llama3_70b/leaderboard.csv`
2. Extracts the model path from each file's location (e.g., `vuyw61pe/w4zu9uh1` from `results/alpaca_eval/dpo/vuyw61pe/w4zu9uh1/.../leaderboard.csv`)
3. Replaces the first column (model name) in each CSV's data row with the extracted path
4. Combines all rows into a single CSV with a shared header

### **Example**

If you have results in:
- `results/alpaca_eval/dpo/1sgwtpjp/v9no9em2/weighted_alpaca_eval_vllm_llama3_70b/leaderboard.csv`
- `results/alpaca_eval/dpo/4nqieq7t/gkzji65z/weighted_alpaca_eval_vllm_llama3_70b/leaderboard.csv`

Running the combine script will create a single CSV with both models' results, where the first column contains `1sgwtpjp/v9no9em2` and `4nqieq7t/gkzji65z` respectively.

---

## Workflow Example

### **1. Evaluate Multiple Models**

For each model, submit a SLURM job:

```bash
# Model 1
sbatch --export=ALL,MODEL_NAME="vuyw61pe/w4zu9uh1",RESULTS_DIR="results/alpaca_eval/dpo" scripts/dpo/run_alpaca_eval.sbatch

# Model 2
sbatch --export=ALL,MODEL_NAME="1sgwtpjp/v9no9em2",RESULTS_DIR="results/alpaca_eval/dpo" scripts/dpo/run_alpaca_eval.sbatch

# ... more models
```

### **2. Wait for Jobs to Complete**

Check job status:
```bash
squeue --me
```

### **3. Combine Results**

After all evaluations complete, combine the leaderboards:

```bash
./scripts/dpo/combine_alpaca_eval_leaderboards.sh
```

The combined leaderboard will be at: `results/alpaca_eval/combined_leaderboard.csv`

---

## Troubleshooting

### **vLLM Server Not Starting**
- Check `annotator_server.log` in the results directory
- Verify the annotator model is available in your HuggingFace cache
- Ensure sufficient GPU memory (try lowering `ANNOTATOR_GPU_MEM_UTILIZATION`)

### **Evaluation Fails**
- Check the SLURM output/error logs in `logs/alpaca_eval/`
- Verify the model exists at `${MODELS_DIR}/${MODEL_NAME}` (default: `models/dpo/${MODEL_NAME}`)
- Check that the custom config file was created correctly at `resources/alpaca_eval/src/alpaca_eval/models_configs/${MODEL_NAME}/configs.yaml`

### **Port Conflicts**
If port 25125 is already in use, change `ANNOTATOR_PORT` to a different value.

---

## Notes

- The script automatically clones the alpaca_eval repository if it doesn't exist
- The script creates a custom config file for each model automatically (no need to create it manually)
- The vLLM server is started once per evaluation and stopped after completion
- Each evaluation runs independently and can be submitted in parallel for multiple models
- Results are saved in a structured directory format: `${RESULTS_DIR}/${MODEL_NAME}/`
- The custom config files are saved in `resources/alpaca_eval/src/alpaca_eval/models_configs/${MODEL_NAME}/configs.yaml` and can be reused for future evaluations of the same model

