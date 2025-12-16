## Code for paper:Towards Near-Optimal Regularization in Deep Learning via the Inverse-Rate Regularizer



### Get Started
install the packges by running  ` pip install -r requirements.txt `



**Single node, single GPU:**

```bash
python -m entry.run1 --conf conf/cifar10.conf -M reg=0 lamb=0.1 sweep_name=cifar10 max_epochs=200
```

**SLURM Job Array: Reg-Lambda Sweep for CIFAR10/CIFAR100 Training**

The `test_array.sh` is a SLURM job array script to run a hyperparameter sweep across combinations of regularization types (`reg`) and lambda values (`lamb`) for training a model on the CIFAR-100 or CIFAR_10 dataset.

---

### 🔧 Script Overview

The `test_array.sh` script is designed for use on a GPU-enabled HPC cluster and schedules **18 jobs** using SLURM’s job array functionality.

Each task corresponds to a unique combination of:

- `reg` values: `3`, `4`, `5`
- `lamb` values: `0.001`, `0.01`, `0.1`, `0.5`, `1.0`, `2.0`

Total combinations: **3 × 6 = 18**

You can change the combination of configuration as you need. Make sure `#SBATCH --array=0-17` corresponding all the sweep job arrays.

---

###  SLURM Resource Configuration
```bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --job-name=Rate_opti
#SBATCH --output=./bin/%A_%a.out
#SBATCH --error=./bin/%A_%a.out
#SBATCH --array=0-17
```

The  `test_ablation.sh`  script is designed for an ablation study that systematically evaluates how model performance changes when the hyperparameters — weight decay (wd) , lambda (lamb) and data augmentation (data_aug) — are varied 

### Features

This codebase adopt configuration file (`.hocon`) to store the hyperparameters (such as the learning rate, training epochs and etc.).
If you want to modify the configuration hyperparameters, you have two ways:

1. Modify the configuration file to generate a new file.

2. You can add `-M` in the running command line to modify the hyperparameters temporarily.


For example, if you hope to modify the total training epochs to 100 and the learning rate to 0.05. You can run the following command:

```bash
python -m entry.run --conf conf/cifar10.conf -o output/cifar10/resnet20 -M max_epochs=100 optimizer.lr=0.05
```

If you modify a non existing hyperparameter, the code will raise an exception.

To list all valid hyperparameters names, you can run the following command:

```bash
pyhocon -i conf/cifar10.conf -f properties
```
`summary_cifar10.csv` and `summary_cifar100.csv` is the summary metric for each model for a specific hyperparameter, it records the metric for the last epoch(200).

`results_cifar10.csv` and `results_cifar100.csv` record the  metric for every model in validation data and test data. 

`val_test_metric.py` generate the results `results_cifar10.csv` and `results_cifar100.csv`, it's input is model after trained for 200 epochs. The models are saved in the folder `output_cifar10` and `output_cifar100`.

`table_validation.py` is used to process `results_cifar10.csv` and `results_cifar100.csv` and generate the latex table in the paper, the generated latex file is in folder `tables`.

the `wandb_L_alphD.py` is the script to generate the csv files to be used in the ablation study plot in the paper. It reads data from ablation study recorded in wandb and it generates the `L.csv` and `alphaD.csv` files.

`L.csv` and `alphaD.csv` is used by the `figure.ipynb` to generate the figure `ablation_ir.pdf` and `ablation_nll.pdf` used in the paper.









---

# 🚀 Experiment Workflow Pipeline

Follow the steps below to process data, generate test metrics, and validate the final results.

---

### 1️⃣ Update Summary Metrics
First, update the summary metrics to ensure the latest training data is captured.
`python update_summary_metric.py` 

### 2️⃣ Export WandB Data
Export the experiment runs from Weights & Biases to a CSV format.
`python wandb_runs_to_csv.py`

### 3️⃣ Generate Test Results
Run the testing script to evaluate model performance on the test set.
`python test_metric.py`

### 4️⃣ Baseline Verification
Check if the baseline data is present in the output CSV.

> **⚠️ Important:** If the baseline is **not** inside the CSV, navigate to the `csv/` folder and run the corresponding Jupyter Notebook (`.ipynb`).

### 5️⃣ Final Validation
Finally, run the validation script to merge train and test data and generate the final result tables.
`python combine_train_test.py`