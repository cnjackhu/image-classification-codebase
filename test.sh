#!/bin/bash --login
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --job-name=Rate_opti
#SBATCH --output=./bin/%A_%a.out
#SBATCH --error=./bin/%A_%a.out
#SBATCH --array=0-3

module purge
eval "$(conda shell.bash hook)"
conda activate torch

# Define the parameter list
PARAM_LIST=(
  "reg=4 lamb=0.1"
  "reg=4 lamb=1.0"
  "reg=8 lamb=0.001"
  "reg=8 lamb=0.01"
)
#Select params for this task
PARAMS="${PARAM_LIST[$SLURM_ARRAY_TASK_ID]}"
echo "Running with $PARAMS"

# Pass them to the Python script
python -m entry.run1 --conf conf/cifar100.conf -o output_41_cifar100_3 -M $PARAMS sweep_name=april1_cifar100 max_epochs=200
