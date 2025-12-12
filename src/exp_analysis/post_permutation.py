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


def plot_metrics_as_axes(metrics, dataset, save_tag='_all'):
    """
    Plot permutation analysis with one figure per permutation type, and one axis per metric.
    Instead of the main function which creates one figure per metric with 4 axes (one per permutation type),
    this creates 4 figures (one per permutation type) with multiple axes (one per metric).
    """
    # Load all permutation results
    noise_type = 'net'
    matrices_net = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    noise_type = 'sign'
    matrices_sign = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    noise_type = 'weight'
    matrices_weight = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    noise_type = 'direction'
    matrices_direction = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    
    # Merge data for all metrics
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
            continue

        merged_df = merge_df(reg_net, reg_sign, reg_weight, reg_dir)
        if merged_df is not None:
            merged_dfs[metric] = merged_df
    
    # Count available metrics
    available_metrics = list(merged_dfs.keys())
    n_metrics = len(available_metrics)
    
    if n_metrics == 0:
        print("No metrics available to plot")
        return
    
    ylabel = "Performance (%)"
    permute_types = ['TF-gene link', 'TF-gene sign', 'TF-gene direction', 'TF-gene weight']
    
    # Helper function to normalize data with min-max scaling to 0-100
    def normalize_minmax(df_all, col):
        """Apply min-max normalization to scale values to 0-100"""
        df = df_all[df_all['Permute type'] == col].drop(columns='Permute type').copy()
        if df.empty:
            return df
        
        df_pivot = df.pivot(index='model', columns='Degree', values='r2score')
        
        # Find global min and max across all models and degrees
        global_min = df_pivot.min().min()
        global_max = df_pivot.max().max()
        
        # Avoid division by zero
        if global_max == global_min:
            df_pivot_norm = df_pivot * 0 + 50  # Set to middle if all values are the same
        else:
            # Min-max normalization: (x - min) / (max - min) * 100
            df_pivot_norm = ((df_pivot - global_min) / (global_max - global_min)) * 100
        
        return df_pivot_norm
    
    # Create one figure per permutation type
    for permute_type in permute_types:
        # Create figure with one axis per metric
        n_cols = min(6, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.7*n_cols, 2.5*n_rows), sharey=False)
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, metric in enumerate(available_metrics):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            df_metric = merged_dfs[metric]
            df_metric_short = df_metric[df_metric['model'].isin(['GRNBoost2', 'Scenic+', 'Pearson Corr.', 'Portia', 'PPCOR'])]
            
            # Normalize the data for this permutation type
            df_pivot_norm = normalize_minmax(df_metric_short, permute_type)
            
            if df_pivot_norm.empty:
                continue
            
            # Plot normalized data
            for model in df_pivot_norm.index:
                ax.plot(df_pivot_norm.columns, df_pivot_norm.loc[model], 
                       label=model, marker='o', color=palette_methods.get(model, 'gray'))
            
            ax.set_ylabel(ylabel if idx == 0 else '')
            ax.margins(x=.1)
            for side in ['right', 'top']:
                ax.spines[side].set_visible(False)
            
            # Only show x-axis label on first axis
            if idx == 0:
                ax.set_xlabel('Permutation intensity (%)')
            else:
                ax.set_xlabel('')
                ax.set_yticks([])
            
            # Wrap long titles: if > 10 characters, replace " (" with " \n("
            title = surrogate_names.get(metric, metric)
            if len(title) > 10:
                title = title.replace(' (', ' \n(')
            ax.set_title(title, pad=12)
            ax.margins(x=.1, y=.2)
            
            # Add legend to the last axis only
            if idx == n_metrics - 1:
                ax.legend(loc=[1.05, .5], frameon=False, fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f"{permute_type}", y=1.02, weight='bold', fontsize=14)
        plt.tight_layout()
        fig_name = f"{figs_dir}/permutation_{permute_type.replace(' ', '_').replace('-', '_')}_{save_tag}.png"
        print(f"Saving figure to: {fig_name}")
        fig.savefig(fig_name, 
                   dpi=200, transparent=True, bbox_inches='tight')