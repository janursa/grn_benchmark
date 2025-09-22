#!/bin/bash
#SBATCH --job-name=metrics_stability
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00  
#SBATCH --mem=250GB
#SBATCH --partition=cpu

source ../env.sh
dataset=$1
gene_wise_output=$2
ws_output=$3
if [ -z "$dataset" ]; then
  echo "Usage: sbatch sbatch.sh <dataset>"
  exit 1
fi
if [ -z "$gene_wise_output" ]; then
  echo "Usage: sbatch sbatch.sh <dataset> <gene_wise_output>"
  exit 1
fi
if [ -z "$ws_output" ]; then
  echo "Usage: sbatch sbatch.sh <dataset> <gene_wise_output> <ws_output>"
  exit 1
fi

echo "Running stability analysis on dataset: $dataset and outputting to: $gene_wise_output and $ws_output"
python src/stability_analysis/metrics_stability/script.py --dataset $dataset --gene_wise_output $gene_wise_output --ws_output $ws_output