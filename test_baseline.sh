#!/bin/sh --login
#SBATCH --time=60:00:00
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

python -m entry.run1 --conf conf/cifar100.conf -M reg=0 lamb=0.1 sweep_name=delete max_epochs=200
# python -m entry.run1 --conf conf/resnet50-tfrec-v1_5.conf -M reg=0 lamb=0.7 \
# sweep_name=imagenet max_epochs=90

# python -m entry.run1 --conf conf/resnet50-tfrec-v1_5.conf -M reg=0 lamb=0.7 \
#     sweep_name=imagenet
