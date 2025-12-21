"""
Evaluate which metrics to keep for each dataset based on:
1. std/mean ratio (coefficient of variation) >= threshold (e.g., 0.2)
2. max score obtained is meaningful based on metric-specific thresholds
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.helper import load_env, surrogate_names

env = load_env()
TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
sys.path.append(f'{TASK_GRN_INFERENCE_DIR}/src/utils/')
from task_grn_inference.src.utils.config import DATASETS, METRICS, METRIC_THRESHOLDS, METRICS_DATASETS



parser = argparse.ArgumentParser(
    description='Evaluate metric quality for each dataset'
)
parser.add_argument(
    '--cv_threshold',
    type=float,
    default=0.2,
    help='Threshold for coefficient of variation (std/mean)'
)
parser.add_argument(
    '--output',
    type=str,
    default=None,
    help='Output CSV file path (optional)'
)
parser.add_argument(
    '--datasets',
    nargs='+',
    default=None,
    help='List of datasets to evaluate (default: all)'
)
parser.add_argument(
    '--metrics',
    nargs='+',
    default=None,
    help='List of metrics to evaluate (default: all METRICS)'
)

parser.add_argument(
    "--local_run",
    action="store_true",
    help="Run in local mode"
)


args = parser.parse_args()
local_run = args.local_run
# print(f"Arguments: {args}")

def load_scores_from_yaml(dataset):
    """Load scores from score_uns.yaml for a given dataset."""
    score_path = Path(f'{TASK_GRN_INFERENCE_DIR}/resources/results/{dataset}/score_uns.yaml')
    
    if not score_path.exists():
        print(f"Warning: {score_path} not found")
        return None
    
    with open(score_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if not data:
        return None
    
    # Convert to DataFrame format: rows = methods, columns = metrics
    scores_dict = defaultdict(dict)
    
    for entry in data:
        if not isinstance(entry, dict):
            continue
        
        method_id = entry.get('method_id')
        metric_ids = entry.get('metric_ids', [])
        metric_values = entry.get('metric_values', [])
        
        if not method_id or not metric_ids:
            continue
        
        # Store metric values for this method
        for metric_id, metric_value in zip(metric_ids, metric_values):
            try:
                scores_dict[method_id][metric_id] = float(metric_value)
            except (ValueError, TypeError):
                continue
    
    if not scores_dict:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(scores_dict, orient='index')
    return df


def evaluate_metric_for_dataset(scores_df, metric, cv_threshold=0.2):
    """
    Evaluate if a metric should be kept for a dataset.
    
    Args:
        scores_df: DataFrame with methods as rows and metrics as columns
        metric: The metric name to evaluate
        cv_threshold: Threshold for coefficient of variation (std/mean)
    
    Returns:
        dict with evaluation results
    """
    if metric not in scores_df.columns:
        return {
            'metric': metric,
            'present': False,
            'reason': 'Metric not computed for this dataset'
        }
    
    values = scores_df[metric].dropna()
    
    if len(values) < 2:
        return {
            'metric': metric,
            'present': True,
            'keep': False,
            'reason': 'Insufficient data points (<2 methods)',
            'n_methods': len(values),
            'mean': np.nan,
            'std': np.nan,
            'cv': np.nan,
            'max': np.nan
        }
    
    mean_val = values.mean()
    std_val = values.std()
    max_val = values.max()
    min_val = values.min()
    
    # Coefficient of variation
    cv = (max_val - min_val) / mean_val if mean_val != 0 else np.inf
    
    # Get threshold for this metric
    threshold = METRIC_THRESHOLDS.get(metric, 0.1)  # Default to 0.1 if not specified
    
    # Decision criteria
    cv_passes = cv >= cv_threshold
    max_passes = max_val >= threshold
    
    keep = cv_passes and max_passes
    
    # Build reason string
    reasons = []
    if not cv_passes:
        reasons.append(f'Low variability (CV={cv:.3f} < {cv_threshold})')
    if not max_passes:
        reasons.append(f'Low max score (max={max_val:.3f} < {threshold})')
    
    if keep:
        reason = 'Passes all criteria'
    else:
        reason = '; '.join(reasons)
    
    return {
        'metric': metric,
        'present': True,
        'keep': keep,
        'reason': reason,
        'n_methods': len(values),
        'mean': mean_val,
        'std': std_val,
        'cv': cv,
        'max': max_val,
        'threshold': threshold,
        'cv_threshold': cv_threshold
    }


def evaluate_all_datasets(datasets=None, metrics=None, cv_threshold=0.2, output_file=None):
    """
    Evaluate all metrics for all datasets.
    
    Args:
        datasets: List of datasets to evaluate (default: all DATASETS)
        metrics: List of metrics to evaluate (default: all METRICS)
        cv_threshold: Threshold for coefficient of variation
        output_file: Path to save results CSV (optional)
    
    Returns:
        DataFrame with evaluation results
    """
    if datasets is None:
        datasets = DATASETS
    if metrics is None:
        metrics = METRICS
    
    all_results = []
    if local_run:
        scores_all = pd.read_csv(f"{TASK_GRN_INFERENCE_DIR}/resources/results/all_scores.csv")
        scores_all = scores_all[METRICS + ['method', 'dataset']]
        scores_all.rename(columns={'method': 'model'}, inplace=True)
    
    # print(f"Evaluating {len(metrics)} metrics across {len(datasets)} datasets")
    # print(f"CV threshold: {cv_threshold}")
    # print(f"\nMetric thresholds:")
    # for metric in metrics:
    #     threshold = METRIC_THRESHOLDS.get(metric, 0.1)
    #     print(f"  {metric}: {threshold}")

    
    for dataset in datasets:
        # print(f"\n{'='*60}")
        # print(f"Processing dataset: {dataset}")
        # print(f"{'='*60}")
        
        # Load scores
        if local_run:
            scores_df = scores_all[scores_all['dataset'] == dataset].drop(columns=['dataset']).set_index('model')
            scores_df = scores_df.loc[:, ~scores_df.isnull().all()]  # Drop columns with all NaNs
        else:
            scores_df = load_scores_from_yaml(dataset)

        
        if scores_df is None:
            # print(f"  No scores found for {dataset}")
            for metric in metrics:
                all_results.append({
                    'dataset': dataset,
                    'metric': metric,
                    'present': False,
                    'keep': False,
                    'reason': 'No scores file found'
                })
            continue
        

        # Evaluate each metric
        for metric in metrics:
            result = evaluate_metric_for_dataset(scores_df, metric, cv_threshold)
            result['dataset'] = dataset
            all_results.append(result)
            
            # Print summary
            if result['present']:
                status = "✓ KEEP" if result.get('keep', False) else "✗ DROP"
                # print(f"  {status} {metric:20s} - {result['reason']}")
            else:
                # print(f"  - SKIP {metric:20s} - {result['reason']}")
                pass
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    cols_order = ['dataset', 'metric', 'keep', 'present', 'reason', 
                  'n_methods', 'mean', 'std', 'cv', 'max', 'threshold', 'cv_threshold']
    cols_order = [c for c in cols_order if c in results_df.columns]
    results_df = results_df[cols_order]
    
    # Save if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}")
    
    # Print summary statistics
    # print(f"\n{'='*60}")
    # print("SUMMARY")
    # print(f"{'='*60}")
    
    # Overall summary
    total_combinations = len(datasets) * len(metrics)
    present_count = results_df['present'].sum()
    keep_count = results_df['keep'].sum()
    
    # print(f"\nTotal dataset-metric combinations: {total_combinations}")
    # print(f"Metrics computed: {present_count} ({100*present_count/total_combinations:.1f}%)")
    # print(f"Metrics to keep: {keep_count} ({100*keep_count/total_combinations:.1f}%)")
    
    # Per dataset summary
    # print(f"\nPer-dataset summary:")
    dataset_present = results_df[results_df['present']].copy()
    # Ensure 'keep' is treated as integer (True=1, False=0)
    dataset_present['keep'] = dataset_present['keep'].astype(int)
    
    dataset_summary = dataset_present.groupby('dataset')['keep'].agg(['sum', 'count'])
    dataset_summary.columns = ['Keep', 'Total']
    # dataset_summary['Drop'] = dataset_summary['Total'] - dataset_summary['Keep']
    dataset_summary['Keep %'] = 100 * dataset_summary['Keep'] / dataset_summary['Total']
    dataset_summary = dataset_summary[['Total', 'Keep', 'Keep %']]
    dataset_summary.sort_values(by='Keep %', ascending=False, inplace=True)
    print(dataset_summary.to_string())
    
    # Per metric summary
    # print(f"\nPer-metric summary:")
    metric_present = results_df[results_df['present']].copy()
    # Ensure 'keep' is treated as integer (True=1, False=0)
    metric_present['keep'] = metric_present['keep'].astype(int)
    metric_summary = metric_present.groupby('metric')['keep'].agg(['sum', 'count'])
    metric_summary.columns = ['Keep', 'Total']
    
    # Consolidate Keep/Total into one column
    metric_summary['Keep/Total'] = metric_summary.apply(
        lambda row: f"{int(row['Keep'])}/{int(row['Total'])}", axis=1
    )
    metric_summary['Keep %'] = 100 * metric_summary['Keep'] / metric_summary['Total']
    
    # Add min/max metric values and CV across datasets
    metric_stats = metric_present.groupby('metric').agg({
        'max': ['min', 'max'],
        'cv': ['min', 'max']
    })
    
    # Consolidate min/max into single columns with rounding
    metric_summary['Value (min/max)'] = metric_stats.apply(
        lambda row: f"{row[('max', 'min')]:.1f}/{row[('max', 'max')]:.1f}", axis=1
    )
    metric_summary['Variability (min/max)'] = metric_stats.apply(
        lambda row: f"{row[('cv', 'min')]:.1f}/{row[('cv', 'max')]:.1f}", axis=1
    )
    
    # Add threshold values for each metric
    metric_thresholds = metric_present.groupby('metric')['threshold'].first()
    metric_summary['Threshold'] = metric_thresholds
    
    # Get valid datasets (where metric was actually computed AND kept)
    metric_kept = metric_present[metric_present['keep'] == 1].copy()
    valid_datasets_dict = metric_kept.groupby('metric')['dataset'].apply(
        lambda x: set(sorted(x.unique()))
    ).to_dict()
    
    # Add additional metadata from METRICS_DATASETS
    from task_grn_inference.src.utils.config import METRICS_DATASETS
  
    
    # Create metadata for each metric group
    metric_group_metadata = {
        'r_precision': {
            "Summary": "Evaluates a GRN by the ability of TFs to predict target gene expression. It only evaluates top regulators.",
            "Applicable Datasets": METRICS_DATASETS.get('regression', []),
            "Dataset Type Required": surrogate_names.get('bulk', 'bulk'),
            "Required Inputs": ", ".join([surrogate_names.get(i, i) for i in ['prediction', 'evaluation_data', 'tf_all', 'regulators_consensus']])
        },
        'r_recall': {
            "Summary": "Evaluates a GRN by the ability of TFs to predict target gene expression. It evaluates broader set of regulators.",
            "Applicable Datasets": METRICS_DATASETS.get('regression', []),
            "Dataset Type Required": surrogate_names.get('bulk', 'bulk'),
            "Required Inputs": ", ".join([surrogate_names.get(i, i) for i in ['prediction', 'evaluation_data', 'tf_all', 'regulators_consensus']])
        },
        'ws_precision': {
            "Summary": "Evaluates a GRN by the shift in gene expression of target gene in response to TF perturbation. It evaluates top TF-edge interactions.",
            "Applicable Datasets": METRICS_DATASETS.get('ws_distance', []),
            "Dataset Type Required": surrogate_names.get('sc', 'sc'),
            "Required Inputs": ", ".join([surrogate_names.get(i, i) for i in ['prediction', 'ws_consensus', 'ws_distance_background']])
        },
        'ws_recall': {
            "Summary": "Evaluates a GRN by the shift in gene expression of target gene in response to TF perturbation. It evaluates broader set of TF-edge interactions.",
            "Applicable Datasets": METRICS_DATASETS.get('ws_distance', []),
            "Dataset Type Required": surrogate_names.get('sc', 'sc'),
            "Required Inputs": ", ".join([surrogate_names.get(i, i) for i in ['prediction', 'ws_consensus', 'ws_distance_background']])
        },
        'sem': {
            "Summary": "Structural Equation Modeling for GRN evaluation.",
            "Applicable Datasets": METRICS_DATASETS.get('sem', []),
            "Dataset Type Required": surrogate_names.get('bulk', 'bulk'),
            "Required Inputs": ", ".join([surrogate_names.get(i, i) for i in ['prediction', 'evaluation_data', 'tf_all']])
        },
        'vc': {
            "Summary": "Evaluate GRNs by their ability in predicting gene expression through neural networks-based virtual cell model.",
            "Applicable Datasets": METRICS_DATASETS.get('vc', []),
            "Dataset Type Required": surrogate_names.get('bulk', 'bulk'),
            "Required Inputs": ", ".join([surrogate_names.get(i, i) for i in ['prediction', 'evaluation_data']])
        },
        't_rec_precision': {
            "Summary": "Measures ability to recover TFs using differential expression and GRN. It only evaluates available TFs in the GRN.",
            "Applicable Datasets": METRICS_DATASETS.get('tf_recovery', []),
            "Dataset Type Required": surrogate_names.get('de', 'de'),
            "Required Inputs": ", ".join([surrogate_names.get(i, i) for i in ['prediction', 'evaluation_data_de', 'tf_all']])
        },
        't_rec_recall': {
            "Summary": "Measures ability to recover TFs using differential expression and GRN. It considers all TFs in the evaluation data.",
            "Applicable Datasets": METRICS_DATASETS.get('tf_recovery', []),
            "Dataset Type Required": surrogate_names.get('de', 'de'),
            "Required Inputs": ", ".join([surrogate_names.get(i, i) for i in ['prediction', 'evaluation_data_de', 'tf_all']])
        },
        'rc_tf_act': {
            "Summary": "Measures consistency of GRN predictions across biological replicates/groups.",
            "Applicable Datasets": METRICS_DATASETS.get('rc_tf_act', []),
            "Dataset Type Required": surrogate_names.get('sc', 'sc'),
            "Required Inputs": ", ".join([surrogate_names.get(i, i) for i in ['prediction', 'evaluation_data']])
        },
        'tfb_f1': {
            "Summary": "Evaluates GRNs by their ability to match Chip-seq TF binding data",
            "Applicable Datasets": METRICS_DATASETS.get('tf_binding', []),
            "Dataset Type Required": surrogate_names.get('bulk', 'bulk'),
            "Required Inputs": ", ".join([surrogate_names.get(i, i) for i in ['prediction', 'evaluation_data', 'Ground Truth']])
        },
        'gs_f1': {
            "Summary": "Evaluates GRNs by their ability to recover known gene sets",
            "Applicable Datasets": METRICS_DATASETS.get('gs_recovery', []),
            "Dataset Type Required": surrogate_names.get('bulk', 'bulk'),
            "Required Inputs": ", ".join([surrogate_names.get(i, i) for i in ['prediction', 'evaluation_data', 'Genesets']])
        }
    }
    
    # Create metadata DataFrame with proper index matching
    metadata_rows = []
    for metric_id in metric_summary.index:
        # Check if metric_id exists directly in metadata
        if metric_id in metric_group_metadata:
            metadata = metric_group_metadata[metric_id].copy()
            
            # Show ALL applicable datasets, but add star only to those that passed threshold
            applicable_ds = set(metadata['Applicable Datasets'])
            valid_ds = valid_datasets_dict.get(metric_id, set())
            
            # Build list with all applicable datasets, marking valid ones with star
            all_datasets_list = []
            for ds in sorted(applicable_ds):
                ds_name = surrogate_names.get(ds, ds)
                if ds in valid_ds:
                    all_datasets_list.append(f"{ds_name}*")
                else:
                    all_datasets_list.append(ds_name)
            
            metadata['Datasets'] = ", ".join(all_datasets_list) if all_datasets_list else ""
            del metadata['Applicable Datasets']
            
            metadata_rows.append(metadata)
        else:
            # Fill with empty values if no metadata found
            valid_ds = valid_datasets_dict.get(metric_id, set())
            datasets_str = ", ".join([f"{surrogate_names.get(ds, ds)}*" for ds in sorted(valid_ds)])
            
            metadata_rows.append({
                "Summary": "",
                "Datasets": datasets_str,
                "Dataset Type Required": "",
                "Required Inputs": ""
            })
    
    metadata_df = pd.DataFrame(metadata_rows, index=metric_summary.index)
    
    # Add metric name as a column for display with surrogate names
    metric_summary_display = metric_summary.copy()
    metric_summary_display.insert(0, 'Metric', [surrogate_names.get(m, m) for m in metric_summary_display.index])
    
    # Select and order final columns: Summary, Inputs, Datasets, then the rest
    final_columns = ['Metric', 'Summary', 'Required Inputs', 'Datasets', 'Threshold', 'Keep/Total', 'Keep %', 'Value (min/max)', 'Variability (min/max)']
    
    # Build the final dataframe
    metric_summary_final = pd.DataFrame()
    metric_summary_final['Metric'] = metric_summary_display['Metric']
    metric_summary_final['Summary'] = metadata_df['Summary'].values
    metric_summary_final['Inputs'] = metadata_df['Required Inputs'].values
    metric_summary_final['Datasets'] = metadata_df['Datasets'].values
    metric_summary_final['Keep/Total'] = metric_summary_display['Keep/Total'].values
    metric_summary_final['Keep %'] = metric_summary_display['Keep %'].values
    metric_summary_final['Value (min/max)'] = metric_summary_display['Value (min/max)'].values
    metric_summary_final['Variability (min/max)'] = metric_summary_display['Variability (min/max)'].values
    metric_summary_final['Threshold'] = metric_summary_display['Threshold'].values
    
    # Sort by Keep %
    metric_summary_final = metric_summary_final.sort_values(by='Keep %', ascending=False)
    
    # Round Keep % to 1 decimal
    metric_summary_final['Keep %'] = metric_summary_final['Keep %'].round(1)

    metric_summary_final = metric_summary_final.drop(columns=['Keep %'])
    
    # Word wrap long text fields for display
    def wrap_text(text, width=40):
        """Wrap text to specified width, preserving full content"""
        if not isinstance(text, str) or len(text) <= width:
            return text
        # Split into lines of max width
        lines = []
        while len(text) > width:
            # Try to break at comma or space
            break_at = text.rfind(',', 0, width)
            if break_at == -1:
                break_at = text.rfind(' ', 0, width)
            if break_at == -1:
                break_at = width
            else:
                break_at += 1  # Include comma/space
            lines.append(text[:break_at].strip())
            text = text[break_at:].strip()
        lines.append(text)
        return '\n'.join(lines)
    
    max_col_width = 40
    for col in ['Summary', 'Inputs', 'Datasets']:
        if col in metric_summary_final.columns:
            metric_summary_final[col] = metric_summary_final[col].apply(
                lambda x: wrap_text(x, max_col_width)
            )
    
    # Print with tabulate for better formatting
    try:
        from tabulate import tabulate
        print("\n" + tabulate(metric_summary_final, headers='keys', tablefmt='grid', showindex=False))
    except ImportError:
        # Fallback to pandas styling if tabulate not available
        print("\n" + metric_summary_final.to_string(index=False))
    
    # Save table as figure
    if output_file:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Manual figure and table sizing - adjust these values as needed
        fig_width = 16
        fig_height = 8
    
        col_widths = [0.12, 0.25, 0.20, 0.20, 0.06, 0.1, 0.1, 0.05]  # Must sum to ~1.0
        row_height = 0.08
        font_size = 7
        
        # Create figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=metric_summary_final.values,
            colLabels=metric_summary_final.columns,
            cellLoc='left',
            loc='center',
            colWidths=col_widths
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        
        # Set row heights
        for i in range(len(metric_summary_final) + 1):  # +1 for header
            for j in range(len(metric_summary_final.columns)):
                table[(i, j)].set_height(row_height)
        
        # Style header
        for i in range(len(metric_summary_final.columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(metric_summary_final) + 1):
            for j in range(len(metric_summary_final.columns)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
        
        # Save figure
        output_path_fig = str(Path(output_file).with_suffix('.png'))
        plt.savefig(output_path_fig, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Table figure saved to: {output_path_fig}")
    
    # Recommended metrics per dataset
    if False:
        print(f"\n{'='*60}")

        for dataset in datasets:
            kept_metrics = results_df[(results_df['dataset'] == dataset) & 
                                    (results_df['keep'] == True)]['metric'].tolist()
            if kept_metrics:
                print(f"\n{dataset}:")
                print(f"  {kept_metrics}")
            else:
                print(f"\n{dataset}: No metrics recommended")
    
    return results_df


def main():
    # Set default output path if not specified
    if args.output is None:
        results_dir = env.get('RESULTS_DIR', str(Path(TASK_GRN_INFERENCE_DIR) / 'resources' / 'results'))
        args.output = f"{results_dir}/metric_quality_evaluation.csv"
    
    # Run evaluation
    results = evaluate_all_datasets(
        datasets=args.datasets,
        metrics=args.metrics,
        cv_threshold=args.cv_threshold,
        output_file=args.output
    )
    
    return results


if __name__ == '__main__':
    main()
