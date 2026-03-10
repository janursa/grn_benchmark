#!/bin/bash
# Process GRN benchmark results: run evaluation, aggregate scores, and plot.
#
# Usage:
#   bash scripts/process_results.sh [--run_evaluation] [--aggregate_scores] [--plot_raw_scores] [--plot_overview]
#
# Flags (all default to false; combine freely):
#   --run_evaluation    Delegate to task_grn_inference local evaluation pipeline (submits SLURM jobs)
#   --aggregate_scores  Collect *_score.h5ad files → resources/results/all_scores.csv
#   --plot_raw_scores   Per-dataset score heatmaps → output/figs/raw_scores/ + docs/source/images/
#   --plot_overview     Summary/overview figure via scripts/create_overview_figure.py
#
# Example – full pipeline:
#   bash scripts/process_results.sh --run_evaluation --aggregate_scores --plot_raw_scores --plot_overview

set -euo pipefail

RUN_EVALUATION=false
AGGREGATE_SCORES=false
PLOT_RAW_SCORES=false
PLOT_OVERVIEW=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_evaluation)   RUN_EVALUATION=true;   shift ;;
        --aggregate_scores) AGGREGATE_SCORES=true; shift ;;
        --plot_raw_scores)  PLOT_RAW_SCORES=true;  shift ;;
        --plot_overview)    PLOT_OVERVIEW=true;     shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Load environment variables (sets TASK_GRN_INFERENCE_DIR, RESULTS_DIR, etc.)
source env.sh

echo "=========================================="
echo "GRN Results Pipeline"
echo "=========================================="
echo "  run_evaluation    : $RUN_EVALUATION"
echo "  aggregate_scores  : $AGGREGATE_SCORES"
echo "  plot_raw_scores   : $PLOT_RAW_SCORES"
echo "  plot_overview     : $PLOT_OVERVIEW"
echo "=========================================="

# ── Step 1: Run GRN evaluation (metric jobs in task_grn_inference) ────────────
if [[ "$RUN_EVALUATION" == "true" ]]; then
    echo ""
    echo "── Step 1: Running GRN evaluation ──"
    cd "$TASK_GRN_INFERENCE_DIR"
    bash scripts/local_workflows/run_grn_evaluation.sh --run_metrics
    cd "$geneRNBI_DIR"
fi

# ── Step 2: Aggregate *_score.h5ad → all_scores.csv ──────────────────────────
if [[ "$AGGREGATE_SCORES" == "true" ]]; then
    echo ""
    echo "── Step 2: Aggregating scores ──"
    python src/process_results/helper.py \
        --temp_dir "${TASK_GRN_INFERENCE_DIR}/output/evaluation" \
        --results_file "${RESULTS_DIR}/all_scores.csv"
fi

# ── Step 3: Plot per-dataset raw score heatmaps ───────────────────────────────
if [[ "$PLOT_RAW_SCORES" == "true" ]]; then
    echo ""
    echo "── Step 3: Plotting raw scores ──"
    python src/process_results/plot.py \
        --scores_file "${RESULTS_DIR}/all_scores.csv" \
        --output_dir_local "${geneRNBI_DIR}/output/figs/raw_scores" \
        --output_dir_docs "${TASK_GRN_INFERENCE_DIR}/docs/source/images"
fi

# ── Step 4: Overview / summary figure ────────────────────────────────────────
if [[ "$PLOT_OVERVIEW" == "true" ]]; then
    echo ""
    echo "── Step 4: Generating overview figure ──"
    python scripts/create_overview_figure.py --local_run
fi

echo ""
echo "Done."
