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
import warnings
from scipy import stats
from grn_benchmark.src.helper import load_env

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
figs_dir = F"{env['RESULTS_DIR']}/figs"

sys.path.append(env['GRN_BENCHMARK_DIR'])
from src.helper import plot_heatmap, surrogate_names, custom_jointplot, palette_celltype, \
                       palette_methods, \
                       palette_datasets, colors_blind, linestyle_methods, palette_datasets, CONTROLS3, linestyle_methods, retrieve_grn_path, \
                        plot_raw_scores
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
args = parser.parse_args()
dataset = args.dataset

sys.path.append(env['TASK_GRN_INFERENCE_DIR'])
from src.utils.config import METRICS
metrics_all = pd.read_csv(f'{RESULTS_DIR}/experiment/global_grns/metrics_{dataset}.csv').fillna(0)
metrics_all['model'] = metrics_all['model'].apply(lambda x: x.replace('.csv', ''))
metrics_all['model'] = metrics_all['model'].apply(lambda x: ':'.join(x.split(':')[:2]) if ':' in x else x)
def extract_tissue_name(x):
    if ':' not in x:
        return x
    method_part = x.split(':')[0]
    tissue_part = x.split(':')[1].strip()
    if tissue_part.lower().startswith('whole blood'):
        tissue_name = 'Blood'
    elif tissue_part.lower().startswith('bone'):
        tissue_name = 'Bone marrow'
    else:
        tissue_name = tissue_part.split()[0]
    return f"{method_part}: {tissue_name}"
metrics_all['model'] = metrics_all['model'].apply(extract_tissue_name)
metrics_all = metrics_all[[c for c in METRICS if c in metrics_all.columns] + ['model']]
metrics_all = metrics_all[~metrics_all['model'].isin(['scenic', 'Scenic'])]

from src.helper import plot_heatmap
fig, ax = plt.subplots(1, 1, figsize=(.7*len(metrics_all.columns), 6), sharey=False)
if 'model' in metrics_all.columns:
    metrics_all.set_index('model', inplace=True)
metrics_all.index = metrics_all.index.map(lambda x: surrogate_names.get(x, x))
metrics_all.columns = metrics_all.columns.map(lambda x: surrogate_names.get(x, x))
plot_heatmap(metrics_all, name='', ax=ax, cmap="viridis")
ax.set_ylabel('GRN models')
file_name = f"{figs_dir}/global_models_{dataset}.png"
print(file_name)
fig.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')