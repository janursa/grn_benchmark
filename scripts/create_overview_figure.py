"""
Create overview figure from combined results.
This script processes the combined trace.csv and score_uns.yaml files,
then calls the R script to generate the summary figure.

This follows the same logic as process_results.ipynb "Overview of performance" section.
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import argparse

# Add grn_benchmark to path and load environment
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.helper import load_env, surrogate_names, read_yaml

env = load_env()
TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
GRN_BENCHMARK_DIR = env['GRN_BENCHMARK_DIR']

# Add task_grn_inference to path
sys.path.insert(0, TASK_GRN_INFERENCE_DIR)
from src.utils.config import DATASETS, METHODS, METRICS, FINAL_METRICS


def process_scores_from_yaml(score_file):
    """Process score_uns.yaml the same way as read_yaml does."""
    with open(score_file, 'r') as f:
        scores_data = yaml.safe_load(f)
    
    # Convert to rows
    rows = []
    for entry in scores_data:
        if entry is None or 'missing' in str(entry):
            continue
        
        dataset_id = entry.get('dataset_id', '')
        method_id = entry.get('method_id', '')
        metric_ids = entry.get('metric_ids', [])
        metric_values = entry.get('metric_values', [])
        
        for metric, value in zip(metric_ids, metric_values):
            if value != "None":
                try:
                    rows.append({
                        'dataset': dataset_id,
                        'model': method_id,
                        'metric': metric,
                        'value': float(value)
                    })
                except (ValueError, TypeError):
                    pass
    
    df = pd.DataFrame(rows)
    # Pivot to get the same format as read_yaml
    scores_all = df.pivot_table(index=['dataset', 'model'], columns='metric', values='value').reset_index()
    
    return scores_all


def process_scores_from_csv(score_file):
    """Process all_scores.csv for local runs."""
    scores_all = pd.read_csv(score_file)
    # Rename 'method' to 'model' to match expected format
    if 'method' in scores_all.columns:
        scores_all = scores_all.rename(columns={'method': 'model'})
    return scores_all


def parse_duration(duration_str):
    """Parse duration string like '10m 20s' or '1h 30m' to hours."""
    if pd.isna(duration_str):
        return 0
    
    duration_str = str(duration_str).strip()
    total_seconds = 0
    
    parts = duration_str.split()
    for part in parts:
        if 'h' in part:
            total_seconds += float(part.replace('h', '')) * 3600
        elif 'm' in part:
            total_seconds += float(part.replace('m', '')) * 60
        elif 's' in part:
            total_seconds += float(part.replace('s', ''))
    
    return total_seconds / 3600


def convert_to_gb(value):
    """Convert memory values to GB."""
    if pd.isna(value):
        return 0
    
    value = str(value).strip()
    unit_to_bytes = {
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "B": 1
    }
    
    try:
        parts = value.split()
        if len(parts) == 2:
            num, unit = parts
            num = float(num)
            unit = unit.upper()
            if unit in unit_to_bytes:
                return num * unit_to_bytes[unit] / (1024**3)
        return float(value)
    except Exception:
        return 0


def process_trace_to_csv(trace_file):
    """
    Process trace file to create a CSV with method as index and resource columns.
    Only uses 'op' dataset traces as requested.
    Specifically captures the actual GRN inference process, not metric evaluations.
    """
    trace_df = pd.read_csv(trace_file, sep='\t')
    
    # The actual GRN inference for op dataset has pattern: (op.METHOD)
    # The metric evaluation has pattern: (METHOD_op.METRIC)
    # We want the inference, not the evaluation!
    
    print(f"   Total trace entries: {len(trace_df)}")
    
    resource_data = []
    
    for method in METHODS:
        # Look for the actual GRN inference process
        # Pattern: run_grn_inference:run_wf:bench:runEachWf:METHOD:processWf:METHOD_process (op.METHOD)
        inference_pattern = f'run_grn_inference.*:{method}:processWf:{method}_process \\(op\\.{method}\\)'
        method_inference = trace_df[trace_df['name'].str.contains(inference_pattern, regex=True, na=False)]
        
        if len(method_inference) == 0:
            # Try alternative patterns
            # Sometimes it might be (op.method) without the full path
            alt_pattern = f'{method}_process \\(op\\.{method}\\)$'
            method_inference = trace_df[trace_df['name'].str.contains(alt_pattern, regex=True, na=False)]
        
        if len(method_inference) == 0:
            print(f"   Warning: No inference trace found for {method}")
            continue
        
        if len(method_inference) > 1:
            print(f"   Warning: Multiple inference traces found for {method}, using first one")
        
        # Get the values from the inference process
        row = method_inference.iloc[0]
        
        duration = parse_duration(row['duration'])
        peak_rss = convert_to_gb(row['peak_rss'])
        
        # Handle CPU usage
        cpu_val = row['%cpu']
        if pd.notna(cpu_val):
            if isinstance(cpu_val, str):
                cpu_usage = float(cpu_val.replace('%', ''))
            else:
                cpu_usage = float(cpu_val)
        else:
            cpu_usage = 0
        
        resource_data.append({
            'method': method,
            'Duration (hour)': duration,
            'Peak memory (GB)': peak_rss,
            'CPU Usage (%)': cpu_usage
        })
        
        print(f"   {method}: duration={duration:.2f}h, memory={peak_rss:.1f}GB, cpu={cpu_usage:.1f}%")
    
    df_resources = pd.DataFrame(resource_data)
    df_resources = df_resources.set_index('method')
    
    # Handle duplicates if any
    if df_resources.index.duplicated().any():
        print(f"   Warning: Found duplicate methods in resources, taking max")
        df_resources = df_resources.groupby(df_resources.index).max()
    
    print(f"\n   Processed resources for {len(df_resources)} methods")
    
    return df_resources


def main(local_run=False, methods=None, datasets=None):
    
    # Get paths from environment
    results_folder = f'{TASK_GRN_INFERENCE_DIR}/resources/results'
    
    combined_dir = Path(results_folder) / 'all_new'
    trace_file = combined_dir / 'trace.csv'
    
    # Choose score file based on mode
    if local_run:
        score_file = combined_dir / 'all_scores.csv'
    else:
        score_file = combined_dir / 'score_uns.yaml'
    
    print("=" * 80)
    print("Creating Overview Figure")
    print("=" * 80)
    print(f"\nUsing base directory: {TASK_GRN_INFERENCE_DIR}")
    print(f"Results folder: {results_folder}")
    print(f"Combined directory: {combined_dir}")
    print(f"Mode: {'LOCAL RUN' if local_run else 'AWS'}")
    print(f"Score file: {score_file}")
    
    # Step 1: Process trace file (only op dataset)
    print("\n1. Processing trace data (op dataset only)...")
    df_res = process_trace_to_csv(trace_file)
    
    # Step 2: Process scores (all datasets)
    print("\n2. Processing scores (all datasets)...")
    if local_run:
        print("   Using local mode: reading from all_scores.csv...")
        scores_all = process_scores_from_csv(score_file)
    else:
        print("   Using AWS mode: reading from score_uns.yaml...")
        scores_all = process_scores_from_yaml(score_file)
    
    # Filter by methods if specified
    if methods is not None:
        scores_all = scores_all[scores_all['model'].isin(methods)]
    
    # Filter by datasets if specified
    if datasets is not None:
        scores_all = scores_all[scores_all['dataset'].isin(datasets)]
    
    print(f"   Loaded scores for {len(scores_all)} method-dataset combinations")
    print(f"   Datasets: {sorted(scores_all['dataset'].unique().tolist())}")
    print(f"   Methods: {sorted(scores_all['model'].unique().tolist())}")
    
    # Step 2.5: Load metrics that passed criteria from metrics_applicibility
    print("\n2.5. Loading metrics that passed applicability criteria...")
    metrics_kept_file = Path(f'{TASK_GRN_INFERENCE_DIR}/resources/results/experiment/metrics_kept_per_dataset.yaml')
    
    if not metrics_kept_file.exists():
        raise FileNotFoundError(f"Metrics kept file not found: {metrics_kept_file}")
    else:
        with open(metrics_kept_file, 'r') as f:
            metrics_kept_per_dataset = yaml.safe_load(f)
        
        # Get metrics that passed for at least one dataset
        all_kept_metrics = set()
        for dataset, metric_list in metrics_kept_per_dataset.items():
            all_kept_metrics.update(metric_list)
        
        print(f"   Loaded metrics that passed criteria: {len(all_kept_metrics)} metrics")
        print(f"   Metrics: {sorted(all_kept_metrics)}")
        
        # Filter FINAL_METRICS to only include those that passed
        final_metrics_filtered = [m for m in FINAL_METRICS if m in all_kept_metrics]
        print(f"   FINAL_METRICS filtered from {len(FINAL_METRICS)} to {len(final_metrics_filtered)} metrics")
        
        # Use filtered metrics for ranking
        FINAL_METRICS_TO_USE = final_metrics_filtered
    
    
    # Step 3: Process scores - EXACT LOGIC FROM NOTEBOOK
    print("\n3. Creating summary dataframe...")
    
    # Get all metrics from METRICS
    metrics = [m for m in scores_all.columns.tolist() if m in METRICS]
    print(f"   Using all METRICS for display: {len(metrics)} metrics")
    
    # Get only FINAL_METRICS for ranking (filtered by applicability)
    final_metrics = [m for m in METRICS if m in FINAL_METRICS_TO_USE]
    print(f"   Using filtered FINAL_METRICS for ranking: {len(final_metrics)} metrics")
    print(f"   Ranking metrics: {final_metrics}")
    
    # Normalize the scores per dataset
    def normalize_scores_per_dataset(df):
        df = df.set_index('model')
        # Mark originally missing values (methods not run on this dataset)
        original_missing = df.isna()
        
        df[df < 0] = 0
        # Normalize each column
        for col in df.columns:
            col_values = df[col]
            col_min = col_values.min()
            col_max = col_values.max()
            if col_max > col_min:
                df[col] = (col_values - col_min) / (col_max - col_min)
            else:
                # All same value - set to 0
                df[col] = 0
        
        # Restore NaN for originally missing values
        df[original_missing] = float('nan')
        return df
    
    df_all_n = scores_all.groupby('dataset').apply(normalize_scores_per_dataset).reset_index()
    
    # Average scores for all datasets (per metric)
    def mean_for_metrics(df):
        # Calculate mean across datasets for each metric, ignoring NaN values
        return df.drop(['dataset'], axis=1).mean(skipna=True)
    
    df_metrics = (
        df_all_n.groupby(['model'])
        .apply(mean_for_metrics)
    )
    
    # Normalize the averaged metrics to [0, 1] range
    for col in df_metrics.columns:
        # Keep track of NaN values
        original_nans = df_metrics[col].isna()
        col_max = df_metrics[col].max()
        col_min = df_metrics[col].min()
        if col_max > col_min:
            df_metrics[col] = (df_metrics[col] - col_min) / (col_max - col_min)
        else:
            df_metrics[col] = 0
        # Restore NaN
        df_metrics.loc[original_nans, col] = float('nan')
    
    # Keep all metrics but reorder: FINAL_METRICS first, then others
    metrics_in_final = [m for m in FINAL_METRICS if m in df_metrics.columns]
    metrics_not_in_final = [m for m in df_metrics.columns if m not in FINAL_METRICS]
    df_metrics = df_metrics[metrics_in_final + metrics_not_in_final]
    
    # Average scores for all datasets (per dataset)
    def mean_for_datasets(df):
        # print(df.set_index('dataset')[metrics].T.mean(skipna=True))
        return df.set_index('dataset')[metrics].T.mean(skipna=True)
    
    df_datasets = (
        df_all_n.groupby(['model'])
        .apply(mean_for_datasets)
        .reset_index()
    )
    df_datasets = df_datasets.pivot(index='model', columns='dataset', values=0)
    
    # Normalize the per-dataset scores to [0, 1] range
    for col in df_datasets.columns:
        original_nans = df_datasets[col].isna()
        col_max = df_datasets[col].max()
        col_min = df_datasets[col].min()
        if col_max > col_min:
            df_datasets[col] = (df_datasets[col] - col_min) / (col_max - col_min)
        else:
            df_datasets[col] = 0
        # Restore NaN for methods not run on this dataset
        df_datasets.loc[original_nans, col] = float('nan')
    
    # Calculate overall scores
    df_scores = pd.concat([df_metrics, df_datasets], axis=1)
    
    # Remove columns (metrics/datasets) that are all NaN across all methods
    cols_before = len(df_scores.columns)
    df_scores = df_scores.dropna(axis=1, how='all')
    cols_after = len(df_scores.columns)
    if cols_before > cols_after:
        print(f"   Removed {cols_before - cols_after} columns that were all NaN")
    
    # Handle any duplicate index entries
    if df_scores.index.duplicated().any():
        print(f"   Warning: Found duplicate methods in scores, taking mean")
        df_scores = df_scores.groupby(df_scores.index).mean()
    
    # Calculate overall score using ONLY FINAL_METRICS
    # This ensures ranking is based only on the final metrics
    final_metrics_cols = [col for col in final_metrics if col in df_scores.columns]
    dataset_cols = [col for col in df_scores.columns if col in DATASETS]
    
    # Average only FINAL_METRICS and dataset scores for overall ranking
    ranking_cols = final_metrics_cols + dataset_cols
    df_scores['overall_score'] = df_scores[ranking_cols].mean(axis=1, skipna=True)
    
    print(f"   Overall scores calculated using only FINAL_METRICS ({len(final_metrics_cols)} metrics) + datasets ({len(dataset_cols)} datasets)")
    
    # Keep track of which methods are in the filtered scores
    methods_in_scores = df_scores.index.tolist()
    
    # Merge scores with resources
    df_summary = pd.concat([df_scores, df_res], axis=1)
    
    # Filter to only include methods that were in the filtered scores
    df_summary = df_summary[df_summary.index.isin(methods_in_scores)]
    
    df_summary = df_summary.fillna(0)
    df_summary.index.name = 'method_name'
    df_summary = df_summary.reset_index()
    
    # Sort by overall score
    df_summary = df_summary.sort_values(by='overall_score', ascending=False)
    
    # Map method names
    df_summary['method_name'] = df_summary['method_name'].map(lambda x: surrogate_names.get(x, x))
    
    # Add user complexity
    df_summary['User-friendly'] = df_summary['method_name'].map({
        'Scenic+': 1,
        'GRNBoost2': 8,
        'Positive Ctrl': 10,
        'Pearson Corr.': 10,
        'Spearman Corr.': 10,
        'CellOracle': 6,
        'Portia': 9,
        'scGLUE': 6,
        'Scenic': 7,
        'FigR': 6,
        'PPCOR': 7,
        'Negative Ctrl': 10,
        'GRaNIE': 6,
        'scPRINT': 5,
        'GeneFormer': 5,
        'scGPT': 3,
    })
    
    df_summary['Complexity'] = df_summary['User-friendly'].max() - df_summary['User-friendly']
    
    # Map column names to surrogate names
    df_summary.columns = [surrogate_names.get(col, col) for col in df_summary.columns]
    
    df_summary = df_summary.fillna(0)
    
    # Add log-transformed resource metrics
    df_summary['memory_log'] = np.log(df_summary['Peak memory (GB)'] + 1)
    df_summary['memory_log'] = np.max(df_summary['memory_log']) - df_summary['memory_log']
    
    df_summary['complexity_log'] = np.log(df_summary['Complexity'] + 1)
    df_summary['complexity_log'] = np.max(df_summary['complexity_log']) - df_summary['complexity_log']
    
    df_summary['duration_log'] = np.log(df_summary['Duration (hour)'] + 1)
    df_summary['duration_log'] = np.max(df_summary['duration_log']) - df_summary['duration_log']
    
    df_summary['duration_str'] = df_summary['Duration (hour)'].round(1).astype(str)
    df_summary['memory_str'] = df_summary['Peak memory (GB)'].round(1).astype(str)
    
    # Step 4: Save summary
    summary_file = f"{results_folder}/summary.tsv"
    df_summary.to_csv(summary_file, sep='\t', index=False)
    print(f"\n4. Saved summary to: {summary_file}")
    print(f"   Summary has {len(df_summary)} methods and {len(df_summary.columns)} columns")
    
    # Step 5: Call R script
    print("\n5. Calling R script to create figure...")
    r_script = f'{GRN_BENCHMARK_DIR}/src/summary_figure.R'
    summary_figure = f"{results_folder}/summary_figure"
    
    import subprocess
    cmd = f"Rscript {r_script} {summary_file} {summary_figure}"
    print(f"   Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"\n✓ Success! Figure saved to:")
        print(f"   - {summary_figure}.pdf")
        print(f"   - {summary_figure}.png")
    else:
        print(f"\n✗ Error running R script:")
        print(result.stderr)
        print("\nStdout:")
        print(result.stdout)
        return 1
    # also save it to docs folder
    DOCS_IMAGES_DIR = Path(env['DOCS_IMAGES_DIR'])
    doc_png = DOCS_IMAGES_DIR / 'summary_figure.png'
    cmd = f"cp {summary_figure}.png {doc_png}"
    subprocess.run(cmd, shell=True)
    print(f"\n✓ Also saved figure to docs folder: {doc_png}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create overview figure from combined results'
    )
    parser.add_argument(
        '--local_run',
        action='store_true',
        help='Use local run mode - read scores from all_scores.csv instead of score_uns.yaml'
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=None,
        help='List of methods to include (space-separated). If not provided, uses METHODS from config.'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='List of datasets to include (space-separated). If not provided, uses DATASETS from config.'
    )
    
    args = parser.parse_args()
    
    # If methods not provided, use METHODS from config
    methods_to_use = args.methods if args.methods else METHODS
    
    # If datasets not provided, use DATASETS from config
    datasets_to_use = args.datasets if args.datasets else DATASETS
    
    exit(main(local_run=args.local_run, methods=methods_to_use, datasets=datasets_to_use))

