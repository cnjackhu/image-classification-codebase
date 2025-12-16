#!/bin/bash
#SBATCH --job-name=imagenet
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=./bin/%A_%a.out
#SBATCH --error=./bin/%A_%a.out
module purge
eval "$(conda shell.bash hook)"
conda activate torch
# Define array of max_norm values
# max_norms=(1 2 3 4 5)

# Extract value for this job
# max_norm=${max_norms[$SLURM_ARRAY_TASK_ID]}
# echo "Running with max_norm=$max_norm"

# Run training script
python -m entry.run_imagenet --conf conf/resnet50.conf \
    -M max_norm= 1.0 \
    sweep_name=im_net max_epochs=90 reg=5 lamb=0.1
