#!/bin/bash
#SBATCH --job-name=sweep_job
#SBATCH --array=0-3
#SBATCH --output=logs/sweep_%a.out
#SBATCH --error=logs/sweep_%a.out
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1

module purge
eval "$(conda shell.bash hook)"
conda activate torch
# Define value arrays
wds=(0.0 0.0005)
data_augs=(true false)

# Index breakdown:
# wd: 2 options (stride 10)
# lamb: 5 options (stride 2)
# data_aug: 2 options (fastest-changing)

# Inner loop (size 2)
aug_idx=$((SLURM_ARRAY_TASK_ID % 2))

# Outer loop (size 2)
wd_idx=$((SLURM_ARRAY_TASK_ID / 2))

wd=${wds[$wd_idx]}
data_aug=${data_augs[$aug_idx]}

# Construct parameter string
PARAMS="wd=$wd data_aug=$data_aug"
echo "Running with $PARAMS"

# Run training script
python -m entry.run1 \
    --conf conf/cifar100.conf \
    -M $PARAMS \
    sweep_name=ablation_nov max_epochs=200 reg=0 lamb=0.1

