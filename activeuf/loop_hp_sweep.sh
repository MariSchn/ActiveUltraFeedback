#!/bin/bash
set -euo pipefail

# Path to your sbatch wrapper (edit if needed)
SBATCH_SCRIPT="./activeuf/active_learning_loop_multi_node.sbatch"

# Dry-run (1 prints commands, 0 actually submits)
DRY_RUN=${DRY_RUN:-1}

# Grids
REG_VALUES=(100 1000) 
EXP_VALUES=(0.9 0.95 0.99) #(0.95 0.975 0.99 0.995 0.999)
OUTER_VALUES=(256 1024) #(32 128)
REPLAY_MULT_VALUES=(100) #(100)
MAX_TRAINING_STEPS=(100)



# Annotator models -> dataset paths (edit if needed)
declare -A DATASET_MAP
# DATASET_MAP[llama70b]="/iopsstor/scratch/cscs/dmelikidze/datasets/combined_annotations_llama"
DATASET_MAP[qwen235b]="/iopsstor/scratch/cscs/dmelikidze/datasets/combined_with_small_qwen_3_235b-features"

# Acquisition functions (strings passed to --acquisition_function_type)
ACQ_FUNCS=(dts infomax)

# Common args passed to the python script (edit or extend)
COMMON_ARGS="--log_kpis --report_to=wandb --use_features"

# Count jobs
num_jobs=0
for reg in "${REG_VALUES[@]}"; do
  for exp in "${EXP_VALUES[@]}"; do
    for outer in "${OUTER_VALUES[@]}"; do
      for mult in "${REPLAY_MULT_VALUES[@]}"; do
        for steps in "${MAX_TRAINING_STEPS[@]}"; do
          for annot in "${!DATASET_MAP[@]}"; do
            for acq in "${ACQ_FUNCS[@]}"; do
              num_jobs=$((num_jobs+1))
            done
          done
        done
      done
    done
  done
done

echo "Preparing to submit ${num_jobs} jobs (grid product). DRY_RUN=${DRY_RUN}"
if [ "${DRY_RUN}" -eq 1 ]; then
  echo "Dry-run mode: commands will be printed, not submitted."
fi

submit_job() {
  local jobname="$1"
  local completions_dataset_path="$2"
  local reg="$3"
  local exp="$4"
  local outer="$5"
  local mult="$6"
  local acq="$7"
  local steps="$8"

  local replay_size=$(( outer * mult ))

  local script_args="--completions_dataset_path ${completions_dataset_path} \
--acquisition_function_type=${acq} \
--regularization_towards_initial_weights=${reg} \
--exponential_decay_base=${exp} \
--outer_loop_batch_size=${outer} \
--replay_buffer_size=${replay_size} \
--max_training_steps=${steps} \
${COMMON_ARGS}"

  # shorten jobname if very long
  if [ ${#jobname} -gt 32 ]; then
    jobname="$(echo ${jobname} | cut -c1-32)"
  fi

  echo "JOB ${jobname}: sbatch --dependency=afterok:1026863 --export=ALL,SCRIPT_ARGS=\"${script_args}\" --job-name=\"${jobname}\" \"${SBATCH_SCRIPT}\""
  if [ "${DRY_RUN}" -eq 0 ]; then
    sbatch --dependency=afterok:1026863 --export=ALL,SCRIPT_ARGS="${script_args}" --job-name="${jobname}" "${SBATCH_SCRIPT}"
  fi
}

# Iterate full Cartesian product
for reg in "${REG_VALUES[@]}"; do
  for exp in "${EXP_VALUES[@]}"; do
    for outer in "${OUTER_VALUES[@]}"; do
      for mult in "${REPLAY_MULT_VALUES[@]}"; do
        for annot in "${!DATASET_MAP[@]}"; do
          for steps in "${MAX_TRAINING_STEPS[@]}"; do
            dataset="${DATASET_MAP[$annot]}"
            for acq in "${ACQ_FUNCS[@]}"; do
              job="sweep_${annot}_${acq}_reg${reg}_exp${exp}_outer${outer}_rx${mult}_steps${steps}"
              submit_job "${job}" "${dataset}" "${reg}" "${exp}" "${outer}" "${mult}" "${acq}" "${steps}"
            done
          done
        done
      done
    done
  done
done

echo "Done. (If DRY_RUN=0, jobs were submitted.)"