import pandas as pd
import numpy as np
import anndata as ad
import tqdm
import json
import warnings
import matplotlib
import sys
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scanpy as sc 
import itertools
import warnings
import os
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from geneRNBI.src.helper import load_env

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
figs_dir = F"{env['RESULTS_DIR']}/figs"

from geneRNBI.src.helper import plot_heatmap, surrogate_names, custom_jointplot, palette_celltype, \
                       palette_methods, \
                       palette_datasets, colors_blind, linestyle_methods, palette_datasets, CONTROLS3, linestyle_methods, retrieve_grn_path, \
                        plot_raw_scores, palette_metrics
from task_grn_inference.src.utils.config import METRICS

DATASETS = ['op', '300BCG', 'ibd_cd', 'ibd_uc', 'norman']
dataset_labels = {'op': 'OPSCA', '300BCG': '300BCG', 'ibd_cd': 'IBD:CD', 'ibd_uc': 'IBD:UC', 'norman': 'Norman'}

def load_scores(dataset):
    df = pd.read_csv(f'{RESULTS_DIR}/experiment/granular_pseudobulk/metrics_{dataset}.csv')
    if 'dataset' in df.columns:
        df = df.drop(columns=['dataset'])
    df = df[[c for c in METRICS if c in df.columns] + ['granularity']]
    df.columns = df.columns.map(lambda name: surrogate_names.get(name, name))
    mask = df['granularity'] == -1
    df.loc[mask, 'granularity'] = np.inf
    df = df.sort_values(by='granularity', ascending=False)
    df.index = df['granularity'].map(lambda name: 'Original' if name == np.inf else name)
    df = df.drop(columns=['granularity'])
    return df

def plot_line_pseudobulking_effect(scores_mat, ax, show_legend=True):
    scores_mat = scores_mat.reset_index()
    df_melted = scores_mat.melt(
        id_vars='granularity', var_name='Method', value_name='Score'
    )

    def normalize_by_original(group):
        original_value = group[group['granularity'] == 'Original']['Score'].values
        if len(original_value) > 0 and original_value[0] != 0:
            group['Normalized Score'] = group['Score'] / original_value[0]
        else:
            group['Normalized Score'] = group['Score']
        return group

    df_melted = df_melted.groupby('Method', group_keys=False).apply(normalize_by_original)
    granularities = sorted([x for x in df_melted['granularity'].unique() if x != 'Original'], reverse=True)
    position_map = {'Original': 0}
    for i, gran in enumerate(granularities, start=1):
        position_map[gran] = i
    df_melted['x_position'] = df_melted['granularity'].map(position_map)

    # Fixed color map keyed by display name so colors are consistent across datasets
    sns.lineplot(
        data=df_melted,
        x='x_position',
        y='Normalized Score',
        hue='Method',
        marker='o',
        ax=ax,
        palette=palette_metrics,
    )
    ax.set_ylabel('Performance \n (relative to single cell)')
    ax.set_xlabel('Clustering granularity \n ( larger = finer )')
    ax.margins(y=0.15)
    xticks = list(range(len(position_map)))
    xtick_labels = ['Single cell'] + [f'Leiden: {int(np.round(float(gran)))}' for gran in granularities]
    ax.set_xticks(xticks)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='center', fontsize=8)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        final_metrics_display = [surrogate_names.get(m, m) for m in METRICS]
        order = []
        for metric in final_metrics_display:
            if metric in labels:
                order.append(labels.index(metric))
        for i, label in enumerate(labels):
            if i not in order:
                order.append(i)
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
        ax.legend(handles, labels, title='Metric', loc=(1.05, 0.2), frameon=False)
    else:
        ax.get_legend().remove()

n = len(DATASETS)
fig, axes = plt.subplots(1, n, figsize=(3 * n + 2, 3), sharey=True)
for i, (dataset, ax) in enumerate(zip(DATASETS, axes)):
    scores_mat = load_scores(dataset)
    show_legend = (i == n - 1)
    plot_line_pseudobulking_effect(scores_mat, ax, show_legend=show_legend)
    ax.set_title(dataset_labels[dataset], fontsize=9)
    if i > 0:
        ax.set_ylabel('')

plt.tight_layout()
file_name = f"{figs_dir}/evaluation_scores_imputation_lineplot.png"
fig.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
print(f"Saved: {file_name}")

# --- Heatmap 1: mean pseudobulked / original, metrics x datasets ---
CAP = 2.0
heatmap_data = {}
for dataset in DATASETS:
    df = pd.read_csv(f'{RESULTS_DIR}/experiment/granular_pseudobulk/metrics_{dataset}.csv')
    if 'dataset' in df.columns:
        df = df.drop(columns=['dataset'])
    metric_cols = [c for c in METRICS if c in df.columns]
    if 'method' in df.columns:
        df_avg = df.groupby('granularity')[metric_cols].mean().reset_index()
    else:
        df_avg = df[metric_cols + ['granularity']]
    df_avg.columns = df_avg.columns.map(lambda name: surrogate_names.get(name, name))
    sc_row = df_avg[df_avg['granularity'] == -1].drop(columns='granularity').iloc[0]
    pseudobulk_rows = df_avg[df_avg['granularity'] != -1].drop(columns='granularity')
    ratio = pseudobulk_rows.mean() / sc_row
    heatmap_data[dataset_labels[dataset]] = ratio

pivot = pd.DataFrame(heatmap_data).T.T
cmap = mcolors.LinearSegmentedColormap.from_list(
    'rg', ['#d73027', '#ffffbf', '#1a9850'])

pivot_display = pivot.clip(upper=CAP)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(pivot_display, ax=ax, cmap=cmap, vmin=0, vmax=CAP,
            annot=pivot.round(2), fmt='.2f', linewidths=0.4, linecolor='white',
            annot_kws={'size': 7},
            cbar_kws={'label': 'Sensitivity', 'shrink': 0.6, 'aspect': 20,
                      'ticks': [0, 1, 2]})
ax.set_xlabel('Dataset')
ax.set_ylabel('')
plt.xticks(rotation=35, ha='right', fontsize=8)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
heatmap_file = f"{figs_dir}/evaluation_scores_pseudobulk_heatmap.png"
fig.savefig(heatmap_file, dpi=300, transparent=True, bbox_inches='tight')
print(f"Saved: {heatmap_file}")

# --- Heatmap 2: mean pseudobulked / original, metrics x methods ---
method_data = {}
for dataset in DATASETS:
    df = pd.read_csv(f'{RESULTS_DIR}/experiment/granular_pseudobulk/metrics_{dataset}.csv')
    if 'dataset' in df.columns:
        df = df.drop(columns=['dataset'])
    if 'method' not in df.columns:
        continue
    metric_cols = [c for c in METRICS if c in df.columns]
    for method, grp in df.groupby('method'):
        sc_row = grp[grp['granularity'] == -1][metric_cols].iloc[0]
        pb_rows = grp[grp['granularity'] != -1][metric_cols]
        ratio = pb_rows.mean() / sc_row
        if method not in method_data:
            method_data[method] = []
        method_data[method].append(ratio)

method_pivot = {}
for method, ratios in method_data.items():
    avg = pd.concat(ratios, axis=1).mean(axis=1)
    avg.index = [surrogate_names.get(c, c) for c in avg.index]
    method_pivot[method] = avg

method_pivot_df = pd.DataFrame(method_pivot)  # metrics x methods
method_pivot_df = method_pivot_df.reindex(
    columns=[m for m in ['pearson_corr', 'portia', 'grnboost'] if m in method_pivot_df.columns])

fig, ax = plt.subplots(figsize=(4, 4))
method_display = method_pivot_df.clip(upper=CAP)
sns.heatmap(method_display, ax=ax, cmap=cmap, vmin=0, vmax=CAP,
            annot=method_pivot_df.round(2), fmt='.2f', linewidths=0.4, linecolor='white',
            annot_kws={'size': 7},
            cbar_kws={'label': 'Sensitivity', 'shrink': 0.6, 'aspect': 20,
                      'ticks': [0, 1, 2]})
ax.set_xlabel('GRN method')
ax.set_ylabel('')
plt.xticks(rotation=35, ha='right', fontsize=8)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
method_heatmap_file = f"{figs_dir}/evaluation_scores_pseudobulk_heatmap_methods.png"
fig.savefig(method_heatmap_file, dpi=300, transparent=True, bbox_inches='tight')
print(f"Saved: {method_heatmap_file}")
