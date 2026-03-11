#!/bin/bash

datasets=(300BCG ibd_cd ibd_uc nakatake norman op parsebioscience replogle xaira_HCT116 xaira_HEK293T)

for dataset in "${datasets[@]}"; do
    sbatch scripts/experiment_causal_directionality.sh $dataset
done
