#!/bin/sh --login
#SBATCH --time=14:00:00
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

# python -m entry.run1 --conf conf/cifar100.conf -M reg=0 lamb=0.1 sweep_name=delete max_epochs=200
# python -m entry.run1 --conf conf/resnet50-tfrec-v1_5.conf -M reg=0 lamb=0.7 \
# sweep_name=imagenet max_epochs=90

# python -m entry.run1 --conf conf/resnet50-tfrec-v1_5.conf -M reg=0 lamb=0.7 \
#     sweep_name=imagenet

# python -m entry.run1 --conf conf/cifar10.conf -M reg=0 lamb=0.1 sweep_name=cifar10_may12 max_epochs=200
# criterion.type_=MultiMarginLoss
# python -m entry.run1 \
#   --conf conf/cifar100.conf -o output_100 -M reg=0 lamb=0.2 sweep_name=cifar100_nov max_epochs=200 

#python -m entry.run1 --conf conf/cifar10.conf -o out -M reg=0 lamb_wd=true lamb=0.1 sweep_name=delete max_epochs=2        #for reg=0

python -m entry.run1 --conf conf/cifar10.conf -o output_10 -M reg=10 lamb_wd=true lamb=0.1 sweep_name=cifar10_sam max_epochs=200     # for reg=10
python -m entry.run1 --conf conf/cifar10.conf -o output_100 -M reg=10 lamb_wd=true lamb=0.1 sweep_name=cifar100_sam max_epochs=200     # for reg=10
 