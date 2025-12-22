#!/bin/bash --login
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=64G
#SBATCH --partition=sbatch
#SBATCH --job-name=code-server
#SBATCH --output=bin/bin/%j.err
#SBATCH --error=bin/bin/%j.err

# use srun to launch Jupyter server in order to reserve a port

srun --resv-ports=1 ./bin/launch-code-server.srun
