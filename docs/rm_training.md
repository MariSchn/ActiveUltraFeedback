# Reward Model Training

This folder contains scripts and configuration files for training a reward model. The training can be executed in **multi-GPU** or **multi-node** settings using `Accelerate` and Slurm (`sbatch`) for distributed training.

---

## **Folder Structure**

### **Key Files**
- **`activeuf/reward_model/training.py`**: The main script for training the reward model.
- **`configs/rm_training.yaml`**: Configuration file for specifying model, dataset, LoRA, and training parameters.
- **`configs/accelerate/multi_node.yaml`**: Configuration file for `Accelerate` to define multi-node training settings.
- **`configs/accelerate/single_node.yaml`**: Configuration file for `Accelerate` to define single-node training settings.
- **`activeuf/reward_model/training.sbatch`**: Slurm batch script for submitting job in multi-node environment.

---

## **Multi-GPU Training**

### **Purpose**
The multi-GPU training setup allows you to efficiently train the reward model across multiple GPUs on a single node.

### **Command**
#### **option 1**
To execute multi-GPU training, use the following command in the terminal:

```bash
accelerate launch --config_file=$SCRATCH/ActiveUltraFeedback/configs/accelerate/multi_node.yaml -m activeuf.reward_model.training --output_dir=$SCRATCH/models/reward_models/my_model --reward_config=$SCRATCH/ActiveUltraFeedback/configs/rm_training.yaml
```

### **Command Breakdown**
- `accelerate launch`: Launches the training script using the `Accelerate` library.
- `--config_file=configs/accelerate/multi_node.yaml`: Specifies the configuration file for multi-node training to Accelerate.
- `-m activeuf.reward_model.training`: The main training script module.
- `--output_dir`: Specifies the directory where the trained model and checkpoints will be saved.
- `--reward_config`: Specifies the path to the reward model configuration file.

---

## **Multi-Node Training**

### **Purpose**
The multi-node training setup allows you to scale training across multiple nodes in a distributed environment.

### **Command**
To execute multi-node training, use the following command:

```bash
sbatch activeuf/reward_model/training.sbatch
```

### **Command Breakdown**
- `sbatch`: Submits the job to the Slurm workload manager.
- `activeuf/reward_model/training.sbatch`: The Slurm batch script for multi-node training.

---

## **Contact**
For any issues or questions, please contact the project maintainers.

---

## **Acknowledgments**
This project uses Hugging Face's `transformers`, `datasets`, and `accelerate` libraries for efficient model training.