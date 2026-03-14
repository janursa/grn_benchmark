#!/usr/bin/env bash
# Post-metric processing pipeline: aggregate scores, combine results, and plot.
# Run this after GRN evaluation (metric calculation) is complete.
#
# Usage:
#   bash scripts/process_results_local.sh

set -euo pipefail

source env.sh

echo "=========================================="
echo "GRN Post-Metric Processing — LOCAL"
echo "=========================================="

echo ""
echo "── Step 1: Aggregating h5ad scores → all_scores.csv ──"
python scripts/aggregate_local_scores.py \
    --temp_dir "${TASK_GRN_INFERENCE_DIR}/output/evaluation" \
    --results_file "${RESULTS_DIR}/all_scores.csv"

echo ""
echo "── Step 2: Combining results → all_new/ ──"
python scripts/combine_results.py

echo ""
echo "── Step 3: Plotting dataset heatmaps ──"
python scripts/plot_raw_scores.py \
    --output_dir_local "${geneRNBI_DIR}/output/figs/raw_scores" \
    --output_dir_docs "${TASK_GRN_INFERENCE_DIR}/docs/source/images"

echo ""
echo "── Step 4: Generating summary figure ──"
python scripts/create_overview_figure.py

echo ""
echo "Done."
