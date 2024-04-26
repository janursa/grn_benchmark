#!/bin/bash
#SBATCH --partition=maxcpu
#SBATCH --time=120:00:00
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




/beegfs/desy/user/nourisaj/R/figr/4.2.3/bin/Rscript run_figR_TF_gene.R

