#!/bin/bash
#SBATCH --job-name=metrics_stability
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00  
#SBATCH --mem=250GB
#SBATCH --partition=cpu

source env.sh

# Initialize variables
dataset=""
gene_wise_output=""
ws_output=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      dataset="$2"
      shift 2
      ;;
    --gene_wise_output)
      gene_wise_output="$2"
      shift 2
      ;;
    --gene_wise_feature_importance)
      gene_wise_feature_importance="$2"
      shift 2
      ;;
    --ws_output)
      ws_output="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: sbatch experiment_metrics_stability.sh --dataset <dataset> [--gene_wise_output <path>] [--ws_output <path>]"
      exit 1
      ;;
  esac
done

# Check required argument
if [ -z "$dataset" ]; then
  echo "Error: --dataset is required"
  echo "Usage: sbatch experiment_metrics_stability.sh --dataset <dataset> [--gene_wise_output <path>] [--ws_output <path>]"
  exit 1
fi

echo "Running stability analysis on dataset: $dataset and outputting to: $gene_wise_feature_importance and $ws_output"

# Build command with only provided arguments
cmd="python src/stability_analysis/metrics_stability/script.py --dataset $dataset"
if [ -n "$gene_wise_feature_importance" ]; then
  cmd="$cmd --gene_wise_feature_importance $gene_wise_feature_importance"
fi
if [ -n "$gene_wise_output" ]; then
  cmd="$cmd --gene_wise_output $gene_wise_output"
fi
if [ -n "$ws_output" ]; then
  cmd="$cmd --ws_output $ws_output"
fi

echo $cmd
# Execute the command
eval $cmd