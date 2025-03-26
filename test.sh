#!/bin/bash --login
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --job-name=Rate_opti
#SBATCH --output=./bin/%A_%a.out
#SBATCH --error=./bin/%A_%a.out
#SBATCH --array=0-6

module purge
eval "$(conda shell.bash hook)"
conda activate torch

# Define the parameter list
PARAM_LIST=(
  "reg=8 lamb=1.0"
  "reg=8 lamb=0.5"
  "reg=8 lamb=0.1"
  "reg=8 lamb=0.01"
  "reg=8 lamb=0.001"
  "reg=8 lamb=0.0001"
  "reg=8 lamb=0.00001"
)
# Select params for this task
PARAMS="${PARAM_LIST[$SLURM_ARRAY_TASK_ID]}"
echo "Running with $PARAMS"

# Pass them to the Python script
python -m entry.run_resnet_1 --conf conf/cifar10.conf -o delete -M $PARAMS
