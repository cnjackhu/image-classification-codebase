#!/bin/bash --login
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --partition=sbatch
#SBATCH --job-name=code-server
#SBATCH --output=bin/bin/%j.err
#SBATCH --error=bin/bin/%j.err

# use srun to launch Jupyter server in order to reserve a port

srun --resv-ports=1 ./bin/launch-code-server.srun
