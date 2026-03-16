#!/usr/bin/env bash
# AWS GRN benchmark results pipeline.
# Merges per-dataset AWS outputs and generates all plots.
#
# Usage:
#   bash scripts/process_results_aws.sh

set -euo pipefail

source env.sh

echo "=========================================="
echo "GRN Results Pipeline — AWS"
echo "=========================================="

echo ""
echo "── Step 1: Combining AWS results → all_new/ ──"
python scripts/combine_results.py --aws_run

echo ""
echo "── Step 2: Evaluating metric applicability per dataset ──"
python scripts/evaluate_metric_applicability.py \
    --output "${RESULTS_DIR}/metric_quality_evaluation.csv"

echo ""
echo "── Step 3: Plotting dataset heatmaps ──"
python scripts/plot_raw_scores.py --aws_run \
    --output_dir_local "${geneRNBI_DIR}/output/figs/raw_scores" \
    --output_dir_docs "${TASK_GRN_INFERENCE_DIR}/docs/source/images"

echo ""
echo "── Step 4: Generating summary figure ──"
python scripts/create_overview_figure.py --aws_run

echo ""
echo "Done."
