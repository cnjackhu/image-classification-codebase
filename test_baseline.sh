#!/bin/sh --login
#SBATCH --time=15:00:00
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

#python -m entry.run1 --conf conf/cifar10.conf -o ouput_41 -M reg=0 lamb=0.1 sweep_name=april1
#max_epochs: 200
python -m entry.run1 --conf conf/cifar100.conf -o output_41_cifar100base -M reg=0 lamb=0.7 sweep_name=april1_cifar100 max_epochs=200
