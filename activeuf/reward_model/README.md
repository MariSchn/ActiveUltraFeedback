# Reward Model Training

This folder contains scripts and configuration files for training a reward model. The training can be executed in **multi-GPU** or **multi-node** settings using `Accelerate` and Slurm (`sbatch`) for distributed training.

---

## **Folder Structure**

### **Key Files**
- **`reward_trainer.py`**: The main script for training the reward model.
- **`reward_config.yaml`**: Configuration file for specifying model, dataset, LoRA, and training parameters.
- **`multi_gpu.yaml`**: Configuration file for `Accelerate` to define multi-GPU training settings.
- **`multi_node.yaml`**: Configuration file for `Accelerate` to define multi-node training settings.
- **`multi_node.sbatch`**: Slurm batch script for submitting job in multi-node environment.
- **`reward_model.py`**: simple interface for loading and working with trained reward models.

---

## **Multi-GPU Training**

### **Purpose**
The multi-GPU training setup allows you to efficiently train the reward model across multiple GPUs on a single node.

### **Command**
To execute multi-GPU training, use the following command in the terminal:

```bash
accelerate launch --config_file=multi_gpu.yaml reward-trainer.py --output_dir=$SCRATCH/reward_model/trainedModels/TrainedModelName --reward_config=$SCRATCH/reward_model/reward-config.yaml
```

### **Command Breakdown**
- `accelerate launch`: Launches the training script using the `Accelerate` library.
- `--config_file=multi_gpu.yaml`: Specifies the configuration file for multi-GPU training to Accelerate.
- `reward-trainer.py`: The main training script.
- `--output_dir=$SCRATCH/reward-model-training2/TrainedModelName`: Specifies the directory where the trained model and checkpoints will be saved.
  - Replace `$SCRATCH/reward-model-training2/TrainedModelName` with your desired output directory.
- `--reward_config=$SCRATCH/reward-model-training2/reward-config.yaml`: Specifies the path to the reward model configuration file.
  - Replace `$SCRATCH/reward-model-training2/reward-config.yaml` with the path to your configuration file.

---

## **Multi-Node Training**

### **Purpose**
The multi-node training setup allows you to scale training across multiple nodes in a distributed environment.

### **Command**
To execute multi-node training, use the following command:

```bash
sbatch -A <project-name> multi-node.sbatch
```

### **Command Breakdown**
- `sbatch`: Submits the job to the Slurm workload manager.
- `-A <project-name>`: Specifies the project account to charge for the job.
  - Replace `<project-name>` with your project account name.
- `multi-node.sbatch`: The Slurm batch script for multi-node training.

---

## **Contact**
For any issues or questions, please contact the project maintainers.

---

## **Acknowledgments**
This project uses Hugging Face's `transformers`, `datasets`, and `accelerate` libraries for efficient model training.