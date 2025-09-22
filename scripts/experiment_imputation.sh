#!/bin/bash
#SBATCH --job-name=imputation
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00
#SBATCH --mem=250GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalil.nourisa@gmail.com

set -euo pipefail

dataset=$1
inference_method=$2
imputation_methods=("original" "knn" "magic")
run_imputation=true
run_grn_inference=true
run_metrics=true

source env.sh
output_dir="${RESULTS_DIR}/experiment/imputation"
output_file="${output_dir}/metrics_${dataset}_${inference_method}.csv"


if [ -z "$dataset" ] || [ -z "$inference_method" ]; then
    echo "Usage: $0 <dataset> <inference_method>"
    exit 1
fi


if [ "$run_imputation" = true ]; then
    echo "Running imputations..."
    python src/stability_analysis/imputation/script.py \
    --dataset "$dataset" \
    --imputation_methods "${imputation_methods[@]}" \
    --output_dir "$output_dir"

fi

if [ "$run_grn_inference" = true ]; then
    echo "Running GRN inference..."
    for imputation in "${imputation_methods[@]}"; do
        rna_file="${output_dir}/${dataset}_${imputation}_rna.h5ad"
        prediction_file="${output_dir}/${dataset}_${imputation}_${inference_method}_prediction.h5ad"

        if [ ! -f "$rna_file" ]; then
            echo "RNA file not found: $rna_file"
            exit 1
        fi


        if [ "$inference_method" == "grnboost" ]; then
            cd $TASK_GRN_INFERENCE_DIR && singularity run $IMAGES_DIR/scenic \
                python "src/methods/grnboost/script.py" \
                --rna "$rna_file" \
                --prediction "$prediction_file"
        else
            echo "Unknown inference method: $inference_method"
            exit 1
        fi
    done
fi

if [ "$run_metrics" = true ]; then
    predictions=""
    for imputation in "${imputation_methods[@]}"; do
        predictions="${predictions} ${output_dir}/${dataset}_${imputation}_${inference_method}_prediction.h5ad"
    done

    echo "Running metrics..."
    score_files=()

    for prediction in $predictions; do
        score_file="${output_dir}/$(basename "${prediction}" .h5ad)_score.h5ad"
        cd "$TASK_GRN_INFERENCE_DIR" && bash src/metrics/all_metrics/run_local.sh \
            --dataset "$dataset" \
            --prediction "$prediction" \
            --score "$score_file"
        score_files+=("$score_file")
    done

    echo "Aggregating results..."
    python - "$output_file" "${score_files[@]}" <<'EOF'
import anndata as ad
import pandas as pd
import sys

output_file = sys.argv[1]
score_files = sys.argv[2:]  # all remaining args are the score files

results_all = []
for f in score_files:
    adata = ad.read_h5ad(f)
    print(adata.uns)
    metrics_keys = adata.uns['metric_ids']
    metrics_values = adata.uns['metric_values']
    df = {k: [v] for k, v in zip(metrics_keys, metrics_values)}
    df = pd.DataFrame(df)

    df["prediction"] = f
    results_all.append(df)

results = pd.concat(results_all, ignore_index=True)
results.to_csv(output_file, index=False)
EOF
fi

echo "Done!"