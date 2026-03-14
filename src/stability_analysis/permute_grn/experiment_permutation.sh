#!/bin/bash
#SBATCH --job-name=permutation
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=2-00:00:00
#SBATCH --mem=250GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

set -e

dataset=$1
if [ -z "$dataset" ]; then
    echo "Usage: $0 <dataset> [--analysis_types type1 type2 ...]"
    exit 1
fi
shift
source env.sh
python src/stability_analysis/permute_grn/script.py --dataset $dataset "$@"
