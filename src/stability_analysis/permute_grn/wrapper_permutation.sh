#!/bin/bash
# Submits experiment_permutation.sh for each dataset listed below.
# Usage: bash src/stability_analysis/permute_grn/wrapper_permutation.sh [dataset1 dataset2 ...]

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

DATASETS=("${@}")

if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(op replogle parsebioscience norman ibd_cd ibd_uc 300BCG nakatake xaira_HCT116 xaira_HEK293T)
fi

for ds in "${DATASETS[@]}"; do
    jid=$(sbatch src/stability_analysis/permute_grn/experiment_permutation.sh "$ds" | awk '{print $4}')
    echo "Submitted $ds → job $jid"
done
