"""
Post-analysis for imputation experiment.
Loads metrics_op.csv and plots relative performance of imputed vs. single-cell GRNs.
One barplot per inference method, showing knn/magic normalized to original (single-cell).

Usage:
    python src/stability_analysis/imputation/post_imputation.py
"""
import os
import sys
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from geneRNBI.src.helper import load_env
env = load_env()

sys.path.insert(0, env['TASK_GRN_INFERENCE_DIR'])
from src.utils.config import METRICS as FINAL_METRICS
from geneRNBI.src.helper import surrogate_names, colors_blind

RESULTS_DIR = env['RESULTS_DIR']
figs_dir = f"{RESULTS_DIR}/figs"
os.makedirs(figs_dir, exist_ok=True)

dataset = 'op'
exp_dir = f"{RESULTS_DIR}/experiment/imputation"

rr = pd.read_csv(f"{exp_dir}/metrics_{dataset}.csv")
rr = rr[['imputation_method', 'inference_method'] + [m for m in FINAL_METRICS if m in rr.columns]]

imputation_mapping = {'original': 'Single-cell', 'magic': 'Magic imput.', 'knn': 'KNN imput.'}
rr['imputation_method'] = rr['imputation_method'].map(imputation_mapping)
rr.columns = ['imputation_method', 'inference_method'] + [
    surrogate_names.get(m, m) for m in FINAL_METRICS if m in pd.read_csv(f"{exp_dir}/metrics_{dataset}.csv").columns
]

metric_cols = [c for c in rr.columns if c not in ('imputation_method', 'inference_method')]

def plot_imput_vs_singlecell(ax, df_method):
    baseline = df_method[df_method['imputation_method'] == 'Single-cell'][metric_cols].iloc[0]
    df_imput = df_method[df_method['imputation_method'] != 'Single-cell'].copy()
    normalized = df_imput[metric_cols].div(baseline).clip(upper=2.0)
    normalized['imputation_method'] = df_imput['imputation_method'].values
    long = normalized.melt(id_vars='imputation_method', var_name='Metric', value_name='value')
    sns.barplot(long, x='imputation_method', y='value', hue='Metric', ax=ax, palette=colors_blind)
    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('')
    ax.set_ylabel('Relative performance\n(imputed / single-cell)')
    ax.margins(x=0.1, y=0.15)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(loc=(1.02, 0.1), title='Metric', frameon=False, fontsize=6, title_fontsize=7)

KEEP_METHODS = ['pearson_corr']
inference_methods = [m for m in rr['inference_method'].unique() if m in KEEP_METHODS]

for method in inference_methods:
    df_method = rr[rr['inference_method'] == method].copy()
    if 'Single-cell' not in df_method['imputation_method'].values:
        print(f"Skipping {method}: no 'original' baseline found")
        continue

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
    plot_imput_vs_singlecell(ax, df_method)
    plt.tight_layout()

    out = f"{figs_dir}/imputation_{method}.png"
    fig.savefig(out, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
