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

def format_permute_grn_results(noise_type='net', metrics=None, dataset=None):
    if metrics is None:
        raise ValueError("You must provide a list of metrics.")
    degrees = [0, 10, 20, 50, 100]
    results = {}
    for degree in degrees:
        rr_file = f"{RESULTS_DIR}/experiment/permute_grn/{dataset}-{noise_type}-{degree}-scores.csv"
        if not os.path.exists(rr_file):
            print(f"File not found: {rr_file}")
            continue
        # Load data
        df = pd.read_csv(
            rr_file,
            index_col=0
        )
        for metric in metrics:
            if metric not in df.columns:
                # print(f"Metric '{metric}' not found in {dataset}-{noise_type}-{degree}-scores.csv")
                continue
            df_metric = df.loc[:, [metric]].copy()
            df_metric['degree'] = degree
            if metric not in results:
                results[metric] = []
            results[metric].append(df_metric)
    metric_matrices = {}
    for metric in results.keys():
        df_metric_all = pd.concat(results[metric], axis=0)
        df_metric_all.index.name = 'model'
        df_metric_all = df_metric_all.reset_index().pivot(
            index="degree", columns="model", values=metric
        )
        metric_matrices[metric] = df_metric_all
    return metric_matrices
    

def merge_df(reg_mat_net=None, reg_mat_sign=None, reg_mat_weight=None, reg_mat_direction=None):
    dfs = []
    type_map = {
        'TF-gene link': reg_mat_net,
        'TF-gene sign': reg_mat_sign,
        'TF-gene weight': reg_mat_weight,
        'TF-gene direction': reg_mat_direction
    }

    # Collect available DataFrames with proper labels
    for label, df in type_map.items():
        if df is not None:
            df = df.copy()
            df['Permute type'] = label
            dfs.append(df)

    if not dfs:
        return None  # no data available

    df_concat = pd.concat(dfs)

    # Optional cleanup
    if 'negative_control' in df_concat.columns:
        df_concat = df_concat.drop(columns=['negative_control'])

    # Rename columns with surrogate names
    df_concat.columns = df_concat.columns.map(lambda name: surrogate_names.get(name, name))

    # Reshape
    df_concat.reset_index(inplace=True)
    df_concat.rename(columns={'degree': 'Degree'}, inplace=True)
    df_concat = df_concat.melt(
        id_vars=['Permute type', 'Degree'],
        var_name='model',
        value_name='r2score'
    )

    # Filter unwanted rows
    df_concat = df_concat[~((df_concat['Permute type'] == 'TF-gene sign') & (df_concat['Degree'] == 100))]

    return df_concat

def plot_line(df_all, ax, col='TF-gene connections', ylabel="R² score decline (%)"):
    def normalize(df):
        df = df.set_index('Degree')
        df['share'] = 0
        baseline = df.loc[0, 'r2score']
        for i, degree in enumerate(df.index):
            if degree == 0:
                share = 0
            else:
                share = 100*np.abs(df.loc[degree, 'r2score'] - baseline)/baseline
                df.loc[degree, 'share'] = share - previous_degree
                
            previous_degree = share
        return df
    df = df_all[df_all['Permute type'] == col].drop(columns='Permute type')
    if False:
        df = df.groupby(['Permute type', 'model']).apply(normalize).reset_index(level='Degree').drop(columns='r2score')
        df['Degree'] = df['Degree'].astype('category')
        df_pivot = df.pivot(index='model', columns='Degree', values='share')
    else:
        df_pivot = df.pivot(index='model', columns='Degree', values='r2score')
    for model in df_pivot.index:
        ax.plot(df_pivot.columns, df_pivot.loc[model], label=model, marker='o', color=palette_methods[model])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_title(col, pad=12)
    ax.margins(x=.1)
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)
def main(metrics, dataset):
    noise_type = 'net'
    matrices_net = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    noise_type = 'sign'
    matrices_sign = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    noise_type = 'weight'
    matrices_weight = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    noise_type = 'direction'
    matrices_direction = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    merged_dfs = {}
    for metric in metrics:
        reg_net = matrices_net.get(metric)
        reg_sign = matrices_sign.get(metric)
        reg_weight = matrices_weight.get(metric)
        reg_dir = matrices_direction.get(metric)

        # Check if all are None or empty
        if not any([
            df is not None and not df.empty
            for df in [reg_net, reg_sign, reg_weight, reg_dir]
        ]):
            # print(f"Skipping metric '{metric}' due to missing all data variants.")
            continue

        merged_df = merge_df(reg_net, reg_sign, reg_weight, reg_dir)
        if merged_df is not None:
            merged_dfs[metric] = merged_df
    for metric in metrics:
        if metric not in merged_dfs:
            # print(f"Skipping metric '{metric}' due to missing data.")
            continue
        # print(metric)
        ylabel = "Performance"
        fig, axes = plt.subplots(1, 4, figsize=(12, 2.5), sharey=False)
        df_metric = merged_dfs[metric]
        df_metric_short = df_metric[df_metric['model'].isin(['GRNBoost2', 'Scenic+', 'Pearson Corr.', 'Portia', 'PPCOR'])]
        ax = axes[0]
        plot_line(df_metric_short, ax, col='TF-gene link', ylabel=ylabel)
        ax.set_xlabel('Permutation intensity (%)')
        ax.margins(y=.2)
        ax.set_xticks([0, 20, 50, 100])
        legend = ax.legend(loc=(1.1, .2), frameon=False)
        ax = axes[1]
        plot_line(df_metric_short, ax, col='TF-gene sign', ylabel=ylabel)
        ax.set_ylabel('')
        ax.margins(y=.2)
        ax.set_xticks([0, 20, 50])
        ax = axes[2]
        plot_line(df_metric_short, ax, col='TF-gene direction', ylabel=ylabel)
        ax.set_ylabel('')
        ax.margins(y=.2)
        ax.set_xticks([0, 20, 50, 100])
        ax = axes[3] 
        plot_line(df_metric_short, ax, col='TF-gene weight', ylabel=ylabel)
        ax.set_ylabel('')
        ax.margins(y=.2)
        ax.set_xticks([0, 20, 50, 100])
        # legend = ax.legend(loc=(1.1, .2), frameon=False)
        plt.suptitle(f"Metric: {surrogate_names.get(metric, metric)}", y=1.05, weight='bold')
        plt.tight_layout()
    # fig.savefig(f"{results_folder}/figs/robustnes_analysis.png", dpi=300, transparent=True, bbox_inches='tight') 