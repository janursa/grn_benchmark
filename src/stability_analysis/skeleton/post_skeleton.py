"""
Post-analysis for skeleton-filtering experiment.
Compares skeleton-filtered GRN scores vs original GRN scores.
Produces a bar plot: x=methods, bars=metrics, y=relative score (skeleton/original).

Usage:
    python src/exp_analysis/post_skeleton.py --dataset op
    python src/exp_analysis/post_skeleton.py --dataset op --all_datasets
"""
import argparse
import os
import sys
import warnings
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from geneRNBI.src.helper import load_env
env = load_env()

sys.path.insert(0, env['geneRNBI_DIR'])
from src.helper import surrogate_names, METHODS, palette_metrics

RESULTS_DIR = env['RESULTS_DIR']
figs_dir = f"{RESULTS_DIR}/figs"
os.makedirs(figs_dir, exist_ok=True)

from task_grn_inference.src.utils.config import METRICS as ALL_METRICS

# tfb_f1 excluded: direction-sensitive by design
# replicate_consistency excluded: measures stability across replicates, not structure
SKIP_METRICS = {'tfb_f1', 'replicate_consistency'}
METRICS_COLS  = [m for m in ALL_METRICS if m not in SKIP_METRICS]

GS_MIN_THRESHOLD = 0.05  # mask gs_f1 cells where original score < threshold


def plot_dataset(dataset: str):
    skeleton_path = f"{RESULTS_DIR}/experiment/skeleton/{dataset}-skeleton-scores.csv"
    all_scores_path = f"{RESULTS_DIR}/all_scores.csv"

    if not os.path.exists(skeleton_path):
        print(f"Skeleton scores not found: {skeleton_path}  — skipping {dataset}")
        return

    scores_skel = pd.read_csv(skeleton_path, index_col=0)

    all_scores = pd.read_csv(all_scores_path)
    scores_orig = (
        all_scores[all_scores["dataset"] == dataset]
        .drop(columns="dataset")
        .set_index("method")
    )

    cols = [c for c in METRICS_COLS if c in scores_skel.columns and c in scores_orig.columns]
    common_methods = scores_skel.index.intersection(scores_orig.index)
    if common_methods.empty:
        print(f"No common methods for {dataset} — skipping")
        return

    scores_skel = scores_skel.loc[common_methods, cols]
    scores_orig = scores_orig.loc[common_methods, cols]

    # ── compute ratio skeleton / original ──────────────────────────────────────
    ratio = (scores_skel / scores_orig).abs()

    # mask near-zero baseline for gs_f1
    if "gs_f1" in ratio.columns and "gs_f1" in scores_orig.columns:
        ratio.loc[scores_orig["gs_f1"] < GS_MIN_THRESHOLD, "gs_f1"] = float("nan")

    # drop unreliable metrics (median baseline too low)
    reliable = scores_orig.median() > 0.01
    cols = [c for c in cols if reliable.get(c, False)]
    ratio = ratio[cols]

    # surrogate names
    ratio.index   = ratio.index.map(lambda x: surrogate_names.get(x, x))
    ratio.columns = ratio.columns.map(lambda x: surrogate_names.get(x, x))

    # order methods by METHODS list
    ordered = list(dict.fromkeys(surrogate_names.get(m, m) for m in METHODS))
    ratio = ratio.reindex([m for m in ordered if m in ratio.index])

    scores_long = (
        ratio.reset_index()
        .melt(id_vars="index", var_name="Metric", value_name="Relative score")
        .rename(columns={"index": "Method"})
    )

    # ── plot ───────────────────────────────────────────────────────────────────
    n_metrics = len(cols)
    palette = {m: palette_metrics[m] for m in scores_long['Metric'].unique() if m in palette_metrics}

    fig, ax = plt.subplots(1, 1, figsize=(max(7, len(ratio) * 0.6), 3.2))
    sns.barplot(scores_long, x="Method", y="Relative score", hue="Metric",
                ax=ax, palette=palette)

    ax.axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.6, label="No change")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Relative score\n(skeleton-filtered / original GRN)")
    ax.set_title(surrogate_names.get(dataset, dataset), weight="bold")
    ax.margins(x=0.05, y=0.15)
    for side in ["right", "top"]:
        ax.spines[side].set_visible(False)
    ax.legend(title="Metric", loc=(1.05, 0.1), frameon=False)

    plt.tight_layout()
    out_path = f"{figs_dir}/skeleton_{dataset}.png"
    fig.savefig(out_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


# ── entry point ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="op")
parser.add_argument("--all_datasets", action="store_true",
                    help="Plot all datasets that have skeleton scores available")
args = parser.parse_args()

if args.all_datasets:
    import glob
    score_files = glob.glob(f"{RESULTS_DIR}/experiment/skeleton/*-skeleton-scores.csv")
    datasets = [os.path.basename(f).replace("-skeleton-scores.csv", "") for f in score_files]
    print(f"Found datasets: {datasets}")
    for ds in datasets:
        plot_dataset(ds)
else:
    plot_dataset(args.dataset)
