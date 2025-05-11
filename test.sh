#!/bin/bash --login
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --job-name=Rate_opti
#SBATCH --output=./bin/%A_%a.out
#SBATCH --error=./bin/%A_%a.out
#SBATCH --array=0-5

module purge
eval "$(conda shell.bash hook)"
conda activate torch

# Define the parameter list
PARAM_LIST=(
  "reg=5 lamb=0.001"
  "reg=5 lamb=0.01"
  "reg=5 lamb=0.1"
  "reg=5 lamb=0.5"
  "reg=5 lamb=1.0"
  "reg=5 lamb=2.0"
)
#Select params for this task
PARAMS="${PARAM_LIST[$SLURM_ARRAY_TASK_ID]}"
echo "Running with $PARAMS"

# Define dataset (choose 'cifar10' or 'cifar100')
dataset="cifar100"

# Get current date: MMDD_HHMM
timestamp=$(date +"%m%d_%H%M")

# Construct the sweep name
sweep_name="${dataset}_${timestamp}"

# Run the command
python -m entry.run1 \
  --conf conf/${dataset}.conf \
  -M $PARAMS \
  sweep_name=$sweep_name \
  max_epochs=200
