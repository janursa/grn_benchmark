"""
Helpers for aggregating metric score files and loading results.
"""

import sys
import argparse
import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path


def aggregate_score_files(temp_dir: str, results_file: str, known_datasets: list[str]) -> pd.DataFrame:
    """
    Collect *_score.h5ad files from temp_dir, parse dataset/method from filename,
    and write a wide-format CSV to results_file.

    Filename convention: {dataset}_{method}_score.h5ad
    """
    scores_dir = Path(temp_dir)
    print(f"Looking for score files in: {scores_dir}")
    print(f"Will save results to: {results_file}")

    all_results = []
    for score_file in sorted(scores_dir.glob("*_score.h5ad")):
        stem = score_file.stem  # e.g. op_grnboost_score
        if stem.endswith("_score"):
            stem = stem[:-6]  # strip '_score'

        dataset, method = None, None
        for ds in known_datasets:
            if stem.startswith(ds + "_"):
                dataset = ds
                method = stem[len(ds) + 1:]
                break

        if not dataset or not method:
            print(f"  Warning: could not parse filename {score_file.name} — skipping")
            continue

        try:
            adata = ad.read_h5ad(score_file)
        except Exception as e:
            print(f"  Error reading {score_file.name}: {e}")
            continue

        metric_ids = adata.uns.get("metric_ids")
        metric_values = adata.uns.get("metric_values")
        if metric_ids is None or metric_values is None:
            print(f"  Warning: no metric data in {score_file.name}")
            continue

        for mid, mval in zip(metric_ids, metric_values):
            try:
                score = float(mval) if isinstance(mval, str) else float(mval)
            except (ValueError, TypeError):
                score = np.nan
            all_results.append({"dataset": dataset, "method": method, "metric": mid, "score": score})

        print(f"  Processed: {dataset} / {method}  ({len(metric_ids)} metrics)")

    if not all_results:
        print("No results found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df_wide = (
        df.pivot_table(index=["dataset", "method"], columns="metric", values="score")
        .reset_index()
    )
    df_wide.columns.name = ""

    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_csv(results_file, index=False)

    print(f"\nResults saved to: {results_file}")
    print(f"  Evaluations : {len(df_wide)}")
    print(f"  Datasets    : {df_wide['dataset'].nunique()}")
    print(f"  Methods     : {df_wide['method'].nunique()}")
    print(f"  Metrics     : {len([c for c in df_wide.columns if c not in ['dataset','method']])}")
    return df_wide


def load_scores(scores_path: str, metrics: list[str], methods: list[str]) -> pd.DataFrame:
    """
    Load all_scores.csv, keep only the requested metrics columns plus
    'method'/'dataset', and filter to the given methods list.
    Renames 'method' → 'model' to match downstream plotting code.
    """
    df = pd.read_csv(scores_path)
    keep_cols = [c for c in metrics if c in df.columns] + ["method", "dataset"]
    df = df[keep_cols]
    df.rename(columns={"method": "model"}, inplace=True)
    df = df[df["model"].isin(methods)]
    return df


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.helper import load_env
    env = load_env()

    task_dir = env["TASK_GRN_INFERENCE_DIR"]
    sys.path.insert(0, task_dir)
    from task_grn_inference.src.utils.config import DATASETS_METRICS, METHODS

    parser = argparse.ArgumentParser(description="Aggregate metric score h5ad files into all_scores.csv")
    parser.add_argument("--temp_dir", default=f"{task_dir}/output/evaluation",
                        help="Directory containing *_score.h5ad files")
    parser.add_argument("--results_file", default=f"{env['RESULTS_DIR']}/all_scores.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    known_datasets = list(DATASETS_METRICS.keys())
    aggregate_score_files(args.temp_dir, args.results_file, known_datasets)


if __name__ == "__main__":
    main()
