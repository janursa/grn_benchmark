#!/bin/bash
#SBATCH --partition=maxcpu
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --job-name=testjob
#SBATCH --output=logs/first-%N-%j.out
#SBATCH --error=logs/first-%N-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jalil.nourisa@gmail.com
unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge
module load maxwell gcc/8.2

for DEGfile in ../../output/infer/ananse/deseq2/*;do touch -m $DEGfile;done

anansnake \
--configfile ../../output/infer/ananse/config.yaml \
--resources mem_mb=60_000 --cores 12

