#!/bin/bash --login
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

module purge
eval "$(conda shell.bash hook)"
conda activate torch

# Define the list of regions and lambda values
regs=(3 4 5)
lambdas=(0.001 0.01 0.1 0.5 1.0 2.0)

# Calculate the indices based on the SLURM_ARRAY_TASK_ID:
# Integer division gives the region index; remainder gives the lambda index.
region=${regs[$((SLURM_ARRAY_TASK_ID / 6))]}
lambda_value=${lambdas[$((SLURM_ARRAY_TASK_ID % 6))]}

PARAMS="reg=$region lamb=$lambda_value"
echo "Running with $PARAMS"

# Pass the parameters to your Python script
#python -m entry.run1 --conf conf/cifar10.conf -o output_41 -M $PARAMS sweep_name=april1 max_epochs=200
#for cifar100
#python -m entry.run1 --conf conf/cifar100.conf -M $PARAMS sweep_name=april2_cifar100 max_epochs=200
python -m entry.run1 \
    --conf conf/cifar100.conf -o output100 -M $PARAMS sweep_name=cifar100_may13 max_epochs=200
