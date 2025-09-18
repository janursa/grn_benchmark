#!/bin/bash
#SBATCH --job-name=imputation
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=60:00:00
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
    python src/stability_analysis/imputation/impute.py \
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

        if [ "$inference_method" == "pearson_corr" ]; then
            cd $TASK_GRN_INFERENCE_DIR && python "src/control_methods/pearson_corr/script.py" \
                --rna "$rna_file" \
                --prediction "$prediction_file"
        elif [ "$inference_method" == "grnboost" ]; then
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
    # Build a space-separated list of model files
    predictions=""
    for imputation in "${imputation_methods[@]}"; do
        predictions="${predictions} ${output_dir}/${dataset}_${imputation}_${inference_method}_prediction.h5ad"
    done

    reg2_consensus_file="${output_dir}/regulators_consensus_${dataset}.json"

    cd "$TASK_GRN_INFERENCE_DIR" && python src/metrics/regression_2/consensus/script.py \
        --dataset "$dataset" \
        --regulators_consensus "$reg2_consensus_file" \
        --evaluation_data "resources/grn_benchmark/evaluation_data/${dataset}_bulk.h5ad" \
        --predictions $predictions


    echo "Running metrics..."
    cd "$GRN_BENCHMARK_DIR" && python src/stability_analysis/imputation/metrics.py \
        --dataset "$dataset" \
        --reg2_consensus_file "$reg2_consensus_file" \
        --predictions $predictions \
        --output_file "$output_file"
fi
echo "Done!"