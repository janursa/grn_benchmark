#!/bin/bash
#SBATCH --job-name=permutation
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00
#SBATCH --mem=250GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

set -e

dataset=$1
if [ -z "$dataset" ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi
source env.sh
python src/stability_analysis/permute_grn/script.py --dataset $dataset
