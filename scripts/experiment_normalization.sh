#!/bin/bash
#SBATCH --job-name=normalization
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2-00:00:00
#SBATCH --mem=250GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalil.nourisa@gmail.com

set -euo pipefail
dataset=$1

if [ -z "$dataset" ]; then
    echo "Usage: $0 <dataset> "
    exit 1
fi

inference_methods=("portia" "pearson_corr" "grnboost" "scenic" "ppcor") # ( "pearson_corr" "grnboost" "scenic" )
layer='pearson_residual'
run_grn_inference=false
run_metrics=true

source env.sh
output_dir="${RESULTS_DIR}/experiment/normalization"
mkdir -p "$output_dir"
output_file="${output_dir}/metrics_${dataset}.csv"

if [ "$run_grn_inference" = true ]; then
    run_type='sbatch'
    for inference_method in "${inference_methods[@]}"; do
        echo "Running GRN inference $inference_method ..."
        rna_file="${INFERENCE_DIR}/${dataset}_rna.h5ad"
        prediction_file="${output_dir}/${dataset}_${inference_method}_prediction.h5ad"

        if [ ! -f "$rna_file" ]; then
            echo "RNA file not found: $rna_file"
            exit 1
        fi

        script_file="src/methods/${inference_method}/run_local.sh"
        
        cd $TASK_GRN_INFERENCE_DIR && $run_type "$script_file" \
            --rna "$rna_file" \
            --prediction "$prediction_file" \
            --layer "$layer"
    done
fi

if [ "$run_metrics" = true ]; then
    script_file="src/metrics/all_metrics/run_local.sh"
    score_files=()
    echo "Run metrics on $layer "
    for inference_method in "${inference_methods[@]}"; do
        prediction="${output_dir}/${dataset}_${inference_method}_prediction.h5ad"
        score_file="${output_dir}/$(basename "${prediction}" .h5ad)_score.h5ad"
        cd "$TASK_GRN_INFERENCE_DIR" && bash src/metrics/all_metrics/run_local.sh \
            --dataset "$dataset" \
            --prediction "$prediction" \
            --score "$score_file" \
            --layer "$layer"
        score_files+=("$score_file")
    done

    echo "Run metrics on default layer"
    for inference_method in "${inference_methods[@]}"; do
        prediction="${RESULTS_DIR}/${dataset}/${dataset}.${inference_method}.${inference_method}.prediction.h5ad"
        score_file="${output_dir}/$(basename "${prediction}" .h5ad)_score_defualt.h5ad"
        cd "$TASK_GRN_INFERENCE_DIR" && bash src/metrics/all_metrics/run_local.sh \
            --dataset "$dataset" \
            --prediction "$prediction" \
            --score "$score_file" \
            --layer "lognorm"
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
    echo "Experiment completed. Results saved to: $output_file"
fi