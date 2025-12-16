#!/bin/bash
#SBATCH --job-name=sweep_job
#SBATCH --array=0-15
#SBATCH --output=logs/sweep_%a.out
#SBATCH --error=logs/sweep_%a.out
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1

module purge
eval "$(conda shell.bash hook)"
conda activate torch
# 1. Define value arrays
wds=(0.0 0.0005)       # Len: 2
lambs=(0.1 1.0 2.0 5.0) # Len: 4
data_augs=(true false)  # Len: 2

# Total combinations: 2 * 4 * 2 = 16 (Matches array 0-15)

# 2. Calculate Indices (The Fix)
# Logic: 
# - data_aug changes every job (fastest)
# - lamb changes every 2 jobs
# - wd changes every 8 jobs (slowest)

# Inner loop (size 2)
aug_idx=$((SLURM_ARRAY_TASK_ID % 2))

# Middle loop (size 4) -> Divide by inner size (2), then mod by middle size (4)
lamb_idx=$(( (SLURM_ARRAY_TASK_ID / 2) % 4 ))

# Outer loop (size 2) -> Divide by (inner * middle) size (2 * 4 = 8)
wd_idx=$((SLURM_ARRAY_TASK_ID / 8))

wd=${wds[$wd_idx]}
lamb=${lambs[$lamb_idx]}
data_aug=${data_augs[$aug_idx]}

# Construct parameter string
PARAMS="wd=$wd lamb=$lamb data_aug=$data_aug"
echo "Running with $PARAMS"

# Run training script
python -m entry.run1 \
    --conf conf/cifar100.conf \
    -M $PARAMS \
    sweep_name=ablation_nov max_epochs=200 reg=5
