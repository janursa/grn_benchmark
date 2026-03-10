"""
Post-analysis: Causal Direction Test
=====================================
Relative performance plot (forward TF→gene / reversed gene→TF), mirroring
the structure of post_tf_masking.py.  Bars = datasets, x = metrics, y = ratio.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from geneRNBI.src.helper import load_env

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']

sys.path.append(env['geneRNBI_DIR'])
from src.helper import surrogate_names, palette_datasets, colors_blind

sys.path.append(env['TASK_GRN_INFERENCE_DIR'])

figs_dir = f"{RESULTS_DIR}/figs/causality_direction"
os.makedirs(figs_dir, exist_ok=True)

# ── load & aggregate scores across models per dataset ─────────────────────────
scores_path = f"{RESULTS_DIR}/experiment/causality_direction/replogle/scores.csv"
df = pd.read_csv(scores_path)

# add dataset column if missing (for backward compat with pre-patch runs)
if 'dataset' not in df.columns:
    df['dataset'] = 'replogle'

metrics_cols = ['ws_precision', 'ws_recall', 'ws_f1']

# Average across models per (dataset, direction)
df_mean = (df.groupby(['dataset', 'direction'])[metrics_cols]
             .mean()
             .reset_index())

# ── compute relative performance (forward / reversed) per dataset ─────────────
datasets = df_mean['dataset'].unique()
relative_data = []

for dataset in datasets:
    fwd = df_mean[(df_mean['dataset'] == dataset) & (df_mean['direction'] == 'forward')]
    rev = df_mean[(df_mean['dataset'] == dataset) & (df_mean['direction'] == 'reversed')]
    if len(fwd) == 0 or len(rev) == 0:
        continue
    for metric in metrics_cols:
        fwd_val = fwd[metric].values[0]
        rev_val = rev[metric].values[0]
        if rev_val != 0 and not np.isnan(fwd_val) and not np.isnan(rev_val):
            relative_data.append({
                'dataset': dataset,
                'metric':  metric,
                'relative_performance': fwd_val / rev_val
            })

df_relative = pd.DataFrame(relative_data)

# apply surrogate names (dataset only; metrics use ws_surrogate below)
df_relative['dataset'] = df_relative['dataset'].map(lambda x: surrogate_names.get(x, x))

# local surrogate names for ws metrics (not in global config)
ws_surrogate = {
    'ws_precision': 'WS Precision',
    'ws_recall':    'WS Recall',
    'ws_f1':        'WS F1',
}
df_relative['metric'] = df_relative['metric'].map(
    lambda x: ws_surrogate.get(x, x))

metrics_plot  = [ws_surrogate[m] for m in metrics_cols]
datasets_plot = df_relative['dataset'].unique()

# ── plot (same structure as post_tf_masking.py) ───────────────────────────────
n_datasets = len(datasets_plot)
# scale width so bars fill ~60% of each metric slot regardless of dataset count
width = 0.6 / max(n_datasets, 1)
fig, ax = plt.subplots(1, 1, figsize=(max(2, 1 + 0.5 * n_datasets), 2.2))

x = np.arange(len(metrics_plot))

for i, dataset in enumerate(datasets_plot):
    dataset_data = df_relative[df_relative['dataset'] == dataset]
    ratios = []
    for metric in metrics_plot:
        row = dataset_data[dataset_data['metric'] == metric]
        ratios.append(row['relative_performance'].values[0] if len(row) > 0 else np.nan)

    # look up original key for palette
    original_dataset = next((d for d in datasets if surrogate_names.get(d, d) == dataset), dataset)
    color = palette_datasets.get(original_dataset, colors_blind[i % len(colors_blind)])

    ax.bar(x + width * i, ratios, width, label=dataset,
           color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='No difference')
ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Relative Performance\n(forward / reversed)', fontsize=12)
ax.set_xticks(x + width * (len(datasets_plot) - 1) / 2)
ax.set_xticklabels(metrics_plot, rotation=45, ha='right')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False,
          fontsize=10, title='Dataset')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ensure bars are visible regardless of whether ratios are above or below 1
all_ratios = df_relative['relative_performance'].dropna().tolist()
if all_ratios:
    y_lo = min(all_ratios + [1.0]) * 0.85
    y_hi = max(all_ratios + [1.0]) * 1.10
    ax.set_ylim(y_lo, y_hi)

file_name = f"{figs_dir}/causality_direction_relative_performance.png"
print(f"Saving figure to: {file_name}")
plt.savefig(file_name, dpi=300, bbox_inches='tight')
plt.close()

