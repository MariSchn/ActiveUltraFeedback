**Goal**: Alpaca Eval with [weighted_alpaca_eval_vllm_llama3_70b](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/weighted_alpaca_eval_vllm_llama3_70b0) annotator

## 2025-12-05 17:00 steps figured out thus far

1. Clone the alpaca_eval repo
  - Run `git clone https://github.com/tatsu-lab/alpaca_eval.git resources/`
  
2. Start an interactive job with the SwissAI OLMES container 
  - Path to sqsh: `/capstor/store/cscs/swissai/infra01/container-images/infra01+ismayilz+olmes+arm64-cuda-root-latest.sqsh`
  - The command that works for me, because I made an `~/.edf/olmes.sqsh`: `srun -A a-infra01-1 --environment olmes --time 01:00:00 --partition debug --pty bash`

3. Start vLLM server for Llama 3.3 70B in the background
  - All parameters are based on the default values in `activeuf.utils.load_model`
  - Change `download-dir` to whatever your HF_CACHE is
  - Make sure you already have Llama 3.3 70B weights in your HF_CACHE
  - Run 
```
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --gpu-memory-utilization 0.9  \
  --swap-space 1  \
  --tensor-parallel-size 4  \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --trust-remote-code  \
  --dtype bfloat16  \
  --port 25125   \
  --api-key token-abc123 \
  --download-dir ../huggingface/ \
> vllm_server.log 2>&1 &
```

4. Wait for the server to be ready, around 4 minutes
  - `vllm_server.log` will say: "INFO:     Application startup complete."

5. `cd` to the alpaca_eval repo
  - Run `cd $SCRATCH/ActiveUltraFeedback/repos/alpaca_eval`

6. Setup environment
  - Create this config file `client_configs/local_configs.yaml` with this content for OpenAI:
```
default:
    - api_key: "token-abc123"
    - base_url: "http://localhost:25125/v1"
```
  - Run `export OPENAI_CLIENT_CONFIG_PATH=client_config/local_configs.yaml` to tell OpenAI about the vLLM server
  - Default `SSL_CERT_FILE` location is something that does not exist in this container, so just delete this env var
    - Run `unset SSL_CERT_FILE`

7. Try making evaluation work
  - Run
```
alpaca_eval evaluate \
    --model_outputs example/outputs.json \
    --annotators_config weighted_alpaca_eval_vllm_llama3_70b
```

Next things to figure out / Things I cannot document fully right now

- In `alpaca_eval/evaluators_configs/weighted_alpaca_eval_vllm_llama3_70b`, I think we should be able to put anything under `completions_kwargs/model_name` because we are passing in precomputed completions anyway (see `example/outputs.json`), but I think needed to change it to a real model path. 
- It's possible I've understood the format of `example/outputs.json` wrongly, and the outputs there are not precomputed completions.
- I was never able to avoid giving OpenAI a real API key. It keeps complaining about authorisation failures.
- Even after giving a real API key (my own), it would run into an authorisation error, albeit a different one. Potentially this is because I forgot to login to huggingface first.