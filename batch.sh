#!/bin/bash
#SBATCH --partition=wgs
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --job-name=testjob
#SBATCH --output=logs/first-%N-%j.out
#SBATCH --error=logs/first-%N-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jalil.nourisa@gmail.com
unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge
module load maxwell gcc/8.2
python scripts/compute_topic_number.py
