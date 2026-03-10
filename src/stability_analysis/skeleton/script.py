"""
Skeleton-filtering experiment.

For each method × dataset:
  1. Load the original GRN prediction (.h5ad)
  2. Filter edges to only those present in the skeleton (TF→gene pairs)
  3. Save filtered GRN to tmp/
  4. Run all_metrics on the filtered GRN
  5. Collect scores alongside the baseline (from all_scores.csv)

Results are written to $RESULTS_DIR/experiment/skeleton/{dataset}-scores.csv
"""

import os
import sys
import glob
import argparse
import pandas as pd
import anndata as ad

env = os.environ

sys.path.insert(0, env['UTILS_DIR'])
sys.path.insert(0, env['METRICS_DIR'])

from util import naming_convention, process_links
from src.params import get_par

sys.path.insert(0, env['geneRNBI_DIR'])

# ------------------------------------------------------------------ helpers --

def load_skeleton(skeleton_path: str) -> set:
    """Return a set of (source, target) tuples from the skeleton CSV."""
    df = pd.read_csv(skeleton_path, usecols=["source", "target"])
    return set(zip(df["source"], df["target"]))


def filter_grn(prediction: pd.DataFrame, skeleton_edges: set) -> pd.DataFrame:
    """Keep only edges whose (source, target) pair exists in the skeleton."""
    mask = prediction.apply(lambda row: (row["source"], row["target"]) in skeleton_edges, axis=1)
    return prediction[mask].reset_index(drop=True)


def main_metrics(par):
    from all_metrics.helper import main as _main_metrics
    return _main_metrics(par)


# --------------------------------------------------------------- main logic --

def main(par):
    os.makedirs(par["write_dir"], exist_ok=True)
    os.makedirs(f"{par['write_dir']}/tmp/", exist_ok=True)

    skeleton_edges = load_skeleton(par["skeleton"])
    print(f"Skeleton loaded: {len(skeleton_edges):,} edges", flush=True)

    df_all = None
    for method in par["methods"]:
        pred_path = f"{par['grns_dir']}/{naming_convention(par['dataset'], method)}"
        if not os.path.exists(pred_path):
            print(f"Skipping {pred_path} (not found)", flush=True)
            continue

        # Load original GRN
        net = ad.read_h5ad(pred_path)
        prediction = pd.DataFrame(net.uns["prediction"])
        prediction = process_links(prediction, par={"max_n_links": 50_000})

        n_before = len(prediction)
        prediction_filtered = filter_grn(prediction, skeleton_edges)
        n_after = len(prediction_filtered)
        print(f"{method}: {n_before} → {n_after} edges after skeleton filtering", flush=True)

        if len(prediction_filtered) == 0:
            print(f"  Skipping {method}: no edges remain after filtering", flush=True)
            continue

        # Write filtered GRN as tmp h5ad
        tmp_path = f"{par['write_dir']}/tmp/{par['dataset']}_{method}_skeleton.h5ad"
        net_filtered = ad.AnnData(
            X=None,
            uns={
                "method_id":  net.uns.get("method_id", method),
                "dataset_id": net.uns.get("dataset_id", par["dataset"]),
                "prediction": prediction_filtered[["source", "target", "weight"]].astype(str),
            },
        )
        net_filtered.write(tmp_path)

        # Run metrics on filtered GRN
        par_eval = {**par, "prediction": tmp_path}
        score = main_metrics(par_eval)
        score.index = [method]

        df_all = score if df_all is None else pd.concat([df_all, score])
        print(f"  Scores computed for {method}", flush=True)

    if df_all is not None:
        out_csv = f"{par['write_dir']}/{par['dataset']}-skeleton-scores.csv"
        df_all.to_csv(out_csv)
        print(f"Saved: {out_csv}", flush=True)
    else:
        print("No scores to save.", flush=True)


# --------------------------------------------------------------- entry point --

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    par = get_par(args.dataset)

    grns_dir = f"{env['RESULTS_DIR']}/{args.dataset}/"
    available_files = glob.glob(f"{grns_dir}/*.prediction.h5ad")
    methods = sorted(set(os.path.basename(f).split(".")[1] for f in available_files))
    print(f"Detected methods for {args.dataset}: {methods}", flush=True)

    skeleton_path = f"{env['PRIOR_DIR']}/skeleton.csv"
    if not os.path.exists(skeleton_path):
        skeleton_path = f"{env['RESULTS_DIR']}/experiment/skeleton/skeleton.csv"
    print(f"Using skeleton: {skeleton_path}", flush=True)

    par = {
        **par,
        "grns_dir":  grns_dir,
        "write_dir": f"{env['RESULTS_DIR']}/experiment/skeleton/",
        "methods":   methods,
        "dataset":   args.dataset,
        "skeleton":  skeleton_path,
    }

    main(par)
