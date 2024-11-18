#!/bin/bash
#SBATCH --job-name=helper_grn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00  
#SBATCH --mem=54GB
#SBATCH --partition=gpu

python test.py