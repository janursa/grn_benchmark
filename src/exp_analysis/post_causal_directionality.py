"""
Post-analysis for causal directionality experiment.
Loads reversed GRN scores (100% edge flip) vs original GRN scores (degree=0)
and produces a bar plot: x=methods, bars=metrics, y=relative score (reversed/original).

Usage:
    python src/exp_analysis/post_causal_directionality.py --dataset replogle
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='replogle')
args = parser.parse_args()
dataset = args.dataset

from task_grn_inference.src.utils.config import METRICS as ALL_METRICS

# tfb_f1 excluded: direction-sensitive by design (reversed GRN = gene→TF, metric expects TF→gene)
# replicate_consistency excluded: measures stability across replicates, not directionality
SKIP_METRICS = {'replicate_consistency'}
METRICS_COLS = [m for m in ALL_METRICS if m not in SKIP_METRICS]

# ── load scores ───────────────────────────────────────────────────────────────
dataset_lower = dataset.lower()
# handle case-sensitive dataset names (e.g., 300BCG)
_exp_dir = f"{RESULTS_DIR}/experiment/causal_directionality"
_candidates = [f for f in os.listdir(_exp_dir) if f.lower() == f"{dataset_lower}-direction-100-scores.csv"]
if _candidates:
    reversed_path = f"{_exp_dir}/{_candidates[0]}"
else:
    reversed_path = f"{_exp_dir}/{dataset_lower}-direction-100-scores.csv"
all_scores_path = f"{RESULTS_DIR}/all_scores.csv"

if not os.path.exists(reversed_path):
    raise FileNotFoundError(f"Reversed scores not found: {reversed_path}")

scores_rev = pd.read_csv(reversed_path, index_col=0)

# Use all_scores.csv as baseline — covers all methods/datasets
all_scores = pd.read_csv(all_scores_path)
scores_orig = all_scores[all_scores['dataset'].str.lower() == dataset_lower].drop(columns='dataset').set_index('method')

# keep only METRICS available in both
cols = [c for c in METRICS_COLS if c in scores_rev.columns and c in scores_orig.columns]
common_methods = scores_rev.index.intersection(scores_orig.index)

scores_rev = scores_rev.loc[common_methods, cols]
scores_orig = scores_orig.loc[common_methods, cols]

# ── compute ratio reversed / original ────────────────────────────────────────
ratio = (scores_rev / scores_orig).abs()

# apply surrogate names
ratio.index = ratio.index.map(lambda x: surrogate_names.get(x, x))
ratio.columns = ratio.columns.map(lambda x: surrogate_names.get(x, x))

# order methods according to METHODS
ordered = [surrogate_names.get(m, m) for m in METHODS]
ordered = list(dict.fromkeys(ordered))  # deduplicate while preserving order
ratio = ratio.reindex([m for m in ordered if m in ratio.index])

# cap for display: bars exceeding CAP are clipped and annotated with 'c'
CAP = 2.0
ratio_capped = ratio.clip(upper=CAP)
capped_mask  = ratio > CAP

scores_long = ratio_capped.reset_index().melt(id_vars='index', var_name='Metric', value_name='Relative score')
scores_long = scores_long.rename(columns={'index': 'Method'})

# ── plot ──────────────────────────────────────────────────────────────────────
n_metrics = len(cols)
fig, ax = plt.subplots(1, 1, figsize=(max(7, len(ratio) * 0.6), 3.2))

palette = {m: palette_metrics[m] for m in scores_long['Metric'].unique() if m in palette_metrics}

sns.barplot(scores_long, x='Method', y='Relative score', hue='Metric',
            ax=ax, palette=palette)

# annotate capped bars with 'c'
for patch, (_, row) in zip(ax.patches, scores_long.iterrows()):
    method = row['Method']
    metric = row['Metric']
    if method in capped_mask.index and metric in capped_mask.columns:
        if capped_mask.loc[method, metric]:
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            ax.text(x, y + 0.03, 'c', ha='center', va='bottom', fontsize=7,
                    color='black', fontweight='bold')

ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.6, label='No change')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_xlabel('')
ax.set_ylabel('Relative score\n(reversed / original GRN)')
ax.set_title(surrogate_names.get(dataset, dataset), weight='bold')
ax.set_ylim(0, CAP * 1.15)
ax.margins(x=0.05)
for side in ['right', 'top']:
    ax.spines[side].set_visible(False)
ax.legend(title='Metric', loc=(1.05, 0.1), frameon=False)

plt.tight_layout()
out_path = f"{figs_dir}/causal_directionality_{dataset}.png"
fig.savefig(out_path, dpi=300, transparent=True, bbox_inches='tight')
print(f"Saved: {out_path}", flush=True)
