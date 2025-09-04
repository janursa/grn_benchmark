#!/bin/bash
#SBATCH --job-name=stability_analysis
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00  
#SBATCH --mem=250GB
#SBATCH --partition=cpu

dataset=$1
output_file=$2
if [ -z "$dataset" ]; then
  echo "Usage: sbatch sbatch.sh <dataset>"
  exit 1
fi
if [ -z "$output_file" ]; then
  echo "Usage: sbatch sbatch.sh <dataset> <output_file>"
  exit 1
fi

echo "Running stability analysis on dataset: $dataset and outputting to: $output_file"
python src/stability_analysis/gene_wise/script.py --dataset $dataset --output_file $output_file