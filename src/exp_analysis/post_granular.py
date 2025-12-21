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
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scanpy as sc 
import itertools
import warnings
import os
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from grn_benchmark.src.helper import load_env

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
figs_dir = F"{env['RESULTS_DIR']}/figs"

sys.path.append(env['GRN_BENCHMARK_DIR'])
from src.helper import plot_heatmap, surrogate_names, custom_jointplot, palette_celltype, \
                       palette_methods, \
                       palette_datasets, colors_blind, linestyle_methods, palette_datasets, CONTROLS3, linestyle_methods, retrieve_grn_path, \
                        plot_raw_scores
TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
sys.path.append(TASK_GRN_INFERENCE_DIR)
from src.utils.config import  FINAL_METRICS
scores_mat = pd.read_csv(f'{RESULTS_DIR}/experiment/granular_pseudobulk/metrics_op.csv').drop(columns=['dataset'])
scores_mat = scores_mat[[c for c in FINAL_METRICS if c in scores_mat.columns] + ['granularity']]
scores_mat.columns = scores_mat.columns.map(lambda name: surrogate_names.get(name, name))
print(scores_mat.head())
mask = scores_mat['granularity'] == -1
scores_mat.loc[mask, 'granularity'] = np.inf
scores_mat = scores_mat.sort_values(by='granularity', ascending=False)


scores_mat.index = scores_mat['granularity']
scores_mat.index = scores_mat.index.map(lambda name: 'Original' if name == np.inf else name)
scores_mat = scores_mat.drop(columns=['granularity'])
fig, ax = plt.subplots(1, 1, figsize=(5, 6), sharey=False)
plot_heatmap(scores_mat, name='', ax=ax, cmap="viridis")

def plot_line_pseudobulking_effect(scores_mat, ax):
    scores_mat = scores_mat.reset_index()
    df_melted = scores_mat.melt(
        id_vars='granularity', var_name='Method', value_name='Score'
    )

    # Normalize based on the original (Single cell) value per method
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
    sns.lineplot(
        data=df_melted,
        x='x_position',
        y='Normalized Score',
        hue='Method',
        marker='o',
        ax=ax,
        palette=colors_blind,
    )
    ax.set_ylabel('Performance \n (relative to single cell)')
    ax.set_xlabel('Clustering granularity \n ( larger = finer )')
    ax.margins(y=0.15)
    handles, labels = ax.get_legend_handles_labels()
    final_metrics_display = [surrogate_names.get(m, m) for m in FINAL_METRICS]
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
    xticks = list(range(len(position_map)))
    xtick_labels = ['Single cell'] + [f'Leiden: {int(np.round(float(gran)))}' for gran in granularities]
    ax.set_xticks(xticks)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='center', fontsize=8)

fig, ax = plt.subplots(1, 1, figsize=(2.5, 2), sharey=False)
plot_line_pseudobulking_effect(scores_mat, ax)
file_name = f"{figs_dir}/evaluation_scores_imputation_lineplot.png"
print(file_name)
fig.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
