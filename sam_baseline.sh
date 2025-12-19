#!/bin/sh --login
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --job-name=Rate_opti
#SBATCH --output=./bin/%j.out
#SBATCH --error=./bin/%j.out

module purge
eval "$(conda shell.bash hook)"
conda activate torch

# --- Task 1: Run on GPU 0 ---
CUDA_VISIBLE_DEVICES=0 python -m entry.run1 \
    --conf conf/cifar10.conf \
    -o output_10 \
    -M reg=10 lamb=0.1 sweep_name=cifar10_sam max_epochs=200 &

# --- Task 2: Run on GPU 1 ---
CUDA_VISIBLE_DEVICES=1 python -m entry.run1 \
    --conf conf/cifar100.conf \
    -o output_100 \
    -M reg=10 lamb=0.1 sweep_name=cifar100_sam max_epochs=200 &

# Wait for both background processes to finish
wait