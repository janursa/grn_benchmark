#!/bin/bash
#SBATCH --job-name=causality_dir
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=2-00:00:00
#SBATCH --mem=350GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalil.nourisa@gmail.com

set -e

DATASET=${1:-replogle}

source env.sh
python src/stability_analysis/causality_direction/script.py --dataset $DATASET
