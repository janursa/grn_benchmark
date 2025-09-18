#!/bin/bash
#SBATCH --job-name=normalization
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
inference_methods=("ppcor" "pearson_corr" "grnboost" "scenic" )
layer='pearson_residual'
run_grn_inference=true
run_metrics=true

source ../env.sh
output_dir="${RESULTS_DIR}/experiment/normalization"
mkdir -p "$output_dir"
output_file="${output_dir}/metrics_${dataset}.csv"

if [ -z "$dataset" ]; then
    echo "Usage: $0 <dataset> "
    exit 1
fi


if [ "$run_grn_inference" = true ]; then
   
    for inference_method in "${inference_methods[@]}"; do
         echo "Running GRN inference $inference_method ..."
        rna_file="${INFERENCE_DIR}/${dataset}_rna.h5ad"
        prediction_file="${output_dir}/${dataset}_${inference_method}_prediction.h5ad"

        if [ ! -f "$rna_file" ]; then
            echo "RNA file not found: $rna_file"
            exit 1
        fi
        
        script_file="src/methods/${inference_method}/script.py"
        command="singularity run $IMAGES_DIR/${inference_method} python"
        if [ "$inference_method" == "pearson_corr" ]; then
            script_file="src/control_methods/${inference_method}/script.py"
            command="python"
        elif [ "$inference_method" == "ppcor" ]; then
            command="singularity run $IMAGES_DIR/${inference_method} Rscript"
            script_file="src/methods/${inference_method}/script.R"
        fi
        

        cd $TASK_GRN_INFERENCE_DIR && $command "$script_file" \
            --rna "$rna_file" \
            --prediction "$prediction_file" \
            --layer "$layer"
    done
fi

if [ "$run_metrics" = true ]; then
    # Build a space-separated list of model files
    predictions=""
    for inference_method in "${inference_methods[@]}"; do
        predictions="${predictions} ${output_dir}/${dataset}_${inference_method}_prediction.h5ad"
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