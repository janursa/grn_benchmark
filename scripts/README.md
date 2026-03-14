### `process_results_local.sh`

Run after local GRN evaluation (independent of Nextflow pipeline) is complete. Aggregates raw h5ad score files, combines results, and generates all figures.

```bash
bash scripts/process_results_local.sh
```

Steps executed (unconditionally, in order):
1. `aggregate_local_scores.py` — `*_score.h5ad` → `$RESULTS_DIR/all_scores.csv`
2. `combine_results.py` — merges per-dataset files into `$RESULTS_DIR/all_new/`
3. `plot_raw_scores.py` — per-dataset score heatmaps
4. `create_overview_figure.py` — summary leaderboard figure

---

### `process_results_aws.sh`

Run after AWS evaluation results have been downloaded (Nextflow pipeline). Reads per-dataset `score_uns.yaml` files instead of h5ad files.

```bash
bash scripts/process_results_aws.sh
```

Steps executed:
1. `combine_results.py --aws_run` — merges per-dataset `score_uns.yaml` into `$RESULTS_DIR/all_new/score_uns.yaml`
2. `plot_raw_scores.py --aws_run` — per-dataset score heatmaps
3. `create_overview_figure.py --aws_run` — summary leaderboard figure

---
