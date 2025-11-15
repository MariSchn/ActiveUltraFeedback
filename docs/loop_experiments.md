## Running loop_experimentations.sh
In order to train dpo models on the multiple active datasets generated via the active learning loop, we can use the loop_experimentations.sh file.

Firstly, we would have to Modify 2 variables there: 
- ```BASE_DATASETS_DIR```
folder where the active datasets have been stored.
- ```BASE_OUTPUT_DIR```
folder where we want our DPO models to be stored.

In case the ```BASE_DATASETS_DIR``` folder contains other dataset as well, or we would want to train DPO models on the subset of the active datasets, we can (and should) modify the ```For Loop``` on line 32 of the script, by adjusting the iteration interval.

The script is adjusted to ignore the datasets, on which the DPO models have already been trained. This is done by checking for the dpo model folder in the specified output directory. This is good in case the job failed to run and the output directory wasn't created. Although the script doens't handle cases when the output folder doesn't contain a valid DPO model (this is because, the dpo script could already be running, and we don't want to overwrite the used folder).

## Running olmes_evals.sh
This file is supposed to be run inside the Swiss-ai Olmes package. 

Git clone the repository from: ```https://github.com/swiss-ai/olmes```

Then add this file in the following directory inside the repository: ```reproducibility-scripts/tulu3_dev/```

To run the script, we would first have to modify the following variables:
- ```DPO_TRAINED_DIR``` directory where we store the models we want to evaluate.
- ```RESULTS_BASE_DIR``` directory where we want the evaluation results to be stored. 
- ```RESULTS_BASE_DIR``` directory where the temporary files will be stored, that are necessary for the olmes package to run the evaluations.

Currently the script evaluates the models according to the following tasks: ```"gsm8k::tulu", "minerva_math::tulu", "ifeval::tulu", "truthfulqa::tulu"```. If you would like to change the configuration, you would have to modify the variable: ```ALL_TASKS``` in the script. 

Similarly to the previous script, we can specify the subset of the models to be evaluated by varying the interval for the ```For Loop``` on line 78.

You may have to adjust the ```./installation/unattended-eval.sh``` script to your working environment. 

## Running display_olmes_results.py
This script generates a .tex file that contains a table of results, that you would have to copy and paste into your tex document. 

You would have to specify the following arguments to the script:
- ```results_directory``` which would be equal to the directory which contains the evaluations done by the olmes package. 
- ```delta``` boolean variable, indicating whether we want to display deltas from the baseline model (such as an SFT model), or just the scores.

the ```baselines``` dictionary contains the scores for the baseline reference models, which can be adjusted to your needs.