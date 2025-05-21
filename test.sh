#!/bin/bash --login
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --job-name=Rate_opti
#SBATCH --output=./bin/%j.out
#SBATCH --error=./bin/%j.out

module purge
eval "$(conda shell.bash hook)"
conda activate torch
# echo "Running CIFAR-10"
python val_test_metric.py cifar10
# echo "finished Running CIFAR-10"
# echo "Running CIFAR-100"
# python val_test_metric.py cifar100
# echo "finished Running CIFAR-100"
