# Scripts

This folder contains scripts to run the GRN benchmark pipeline — from results processing and plotting to stability/sensitivity experiments. All scripts source `env.sh` from the repo root to set up paths. SLURM scripts (`experiment_*.sh`) can be submitted via `sbatch` or run directly with `bash`.

---

## `process_results.sh`

Orchestrates the full results pipeline from evaluation through visualization. Run after GRN inference is complete.

```bash
bash scripts/process_results.sh [--run_evaluation] [--aggregate_scores] [--plot_raw_scores] [--plot_overview]
```

| Flag | Description |
|---|---|
| `--run_evaluation` | Delegates to `task_grn_inference` to submit SLURM metric evaluation jobs |
| `--aggregate_scores` | Collects `*_score.h5ad` files → `$RESULTS_DIR/all_scores.csv` |
| `--plot_raw_scores` | Plots per-dataset score heatmaps → `output/figs/raw_scores/` and `task_grn_inference/docs/source/images/` |
| `--plot_overview` | Generates the summary/leaderboard overview figure |

Flags can be combined freely. Typical full run:
```bash
bash scripts/process_results.sh --aggregate_scores --plot_raw_scores --plot_overview
```

---

## `combine_results.py`

Combines trace files, score YAML files, and dataset metadata from individual per-dataset result folders into a single unified directory (`resources/results/all_new/`). Supports two modes:

- **Local mode** (`--local_run`): copies `all_scores.csv` directly
- **AWS mode**: reads individual `score_uns.yaml` files from each dataset folder

```bash
python scripts/combine_results.py --local_run
```

---

## `create_overview_figure.py`

Generates the summary leaderboard figure from combined results. Reads scores, ranks methods across datasets, and calls the R script (`src/summary_figure.R`) to produce the final publication-quality figure.

```bash
python scripts/create_overview_figure.py --local_run
python scripts/create_overview_figure.py --methods grnboost ppcor ... --datasets op 300BCG ...
```

---

## `experiment_normalization.sh`

Compares GRN inference quality under different data normalization strategies (e.g. `pearson_residual` vs default `lognorm`). Runs inference and metrics for a set of methods on a given dataset, saving results to `output/experiment/normalization/`.

```bash
sbatch scripts/experiment_normalization.sh <dataset>
```

---

## `experiment_permutation.sh`

Tests metric robustness by running evaluation on permuted (shuffled) GRN predictions. Provides a null distribution baseline to assess whether observed scores are above chance.

```bash
sbatch scripts/experiment_permutation.sh <dataset>
```

---

## `experiment_causality.sh`

Evaluates how well inferred GRNs capture causal directionality using the causality analysis module (`src/stability_analysis/causality/`). Runs on all datasets defined in the environment.

```bash
sbatch scripts/experiment_causality.sh
```

---

## `experiment_causality_direction.sh`

Extends the causality analysis by testing directionality (source→target vs target→source) for a specific dataset. More compute-intensive than `experiment_causality.sh`.

```bash
sbatch scripts/experiment_causality_direction.sh <dataset>
```

---

## `experiment_causal_directionality.sh`

Analyses causal directionality patterns across predicted GRNs for a given dataset, using a different analytical approach than `experiment_causality_direction.sh` (see `src/stability_analysis/causal_directionality/`).

```bash
sbatch scripts/experiment_causal_directionality.sh <dataset>
```

---

## `experiment_global_grns.sh`

Compares cell-type-specific GRNs against global (non-cell-type-stratified) GRNs, assessing whether cell-type granularity improves benchmark scores.

```bash
sbatch scripts/experiment_global_grns.sh <dataset>
```

---

## `experiment_granular.sh`

Examines how pseudobulk granularity (number of pseudobulk samples, cell count per sample) affects inference quality. Uses the pseudobulk granularity module (`src/stability_analysis/pseudobulk/granularity/`).

```bash
sbatch scripts/experiment_granular.sh
```

---

## `experiment_sc_vs_bulk.sh`

Compares GRN inference quality when using single-cell data directly versus pseudobulk-aggregated data as the inference input.

```bash
sbatch scripts/experiment_sc_vs_bulk.sh
```

---

## `experiment_imputation.sh`

Tests the effect of gene expression imputation methods (none, KNN, MAGIC) on GRN inference quality. Runs inference and evaluation for each imputation strategy on a given dataset.

```bash
sbatch scripts/experiment_imputation.sh <dataset>
```

---

## `experiment_gb_vs_ridge.sh`

Benchmarks Gradient Boosting vs Ridge regression as the underlying model for regression-based GRN metrics. Compares score profiles across inference methods and datasets.

```bash
sbatch scripts/experiment_gb_vs_ridge.sh <dataset> [GB|ridge]
```

---

## `experiment_ensemble.sh`

Evaluates ensemble GRN strategies — combining predictions from multiple inference methods — and scores the result against individual methods.

```bash
sbatch scripts/experiment_ensemble.sh [args]
```

---

## `experiment_skeleton.sh`

Assesses the effect of using a prior GRN skeleton (e.g. from ChIP-seq or ATAC-seq) to constrain the inference search space. Compares constrained vs unconstrained performance.

```bash
sbatch scripts/experiment_skeleton.sh <dataset>
```

---

## `experiment_metrics_stability.sh`

Evaluates the stability of individual metrics across repeated subsampling or bootstrapping of the evaluation data. Produces gene-wise and Wasserstein-distance stability outputs.

```bash
sbatch scripts/experiment_metrics_stability.sh --dataset <dataset> ...
```

---

## `jobs.sh`

A convenience launcher script that submits a predefined set of experiments to SLURM in one go. Edit this file to configure which experiments to run and on which datasets before submitting.

```bash
bash scripts/jobs.sh
```
