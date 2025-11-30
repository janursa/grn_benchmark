"""
Combine trace files, score_uns.yaml, and dataset_uns.yaml from individual dataset result folders.
"""

import os
import yaml
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import sys

# Add paths for imports
sys.path.append('/home/jnourisa/projs/ongoing/task_grn_inference/src/utils/')
from task_grn_inference.src.utils.config import DATASETS

def combine_results():
    """Combine results from individual dataset folders into all_new folder."""
    
    base_dir = Path('/home/jnourisa/projs/ongoing/task_grn_inference/resources/results')
    save_dir = base_dir / 'all_new'
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Processing {len(DATASETS)} datasets: {DATASETS}")
    
    # 1. Combine trace_merged.txt files
    print("\n1. Combining trace_merged.txt files...")
    all_traces = []
    
    for dataset in DATASETS:
        trace_path = base_dir / dataset / 'trace_merged.txt'
        
        if not trace_path.exists():
            print(f"  Warning: {trace_path} not found, skipping...")
            continue
        
        print(f"  Reading trace from {dataset}...")
        df = pd.read_csv(trace_path, sep='\t')
        all_traces.append(df)
    
    if all_traces:
        # Combine all traces and remove duplicates
        combined_traces = pd.concat(all_traces, ignore_index=True)
        combined_traces = combined_traces.drop_duplicates(subset=['name'])
        
        # Save combined trace
        output_trace = save_dir / 'trace.csv'
        combined_traces.to_csv(output_trace, sep='\t', index=False)
        print(f"  Saved combined trace to {output_trace}")
        print(f"  Total unique entries: {len(combined_traces)}")
    else:
        print("  No trace files found!")
    
    # 2. Combine score_uns.yaml files
    print("\n2. Combining score_uns.yaml files...")
    merged_scores = []
    
    for dataset in DATASETS:
        score_path = base_dir / dataset / 'score_uns.yaml'
        
        if not score_path.exists():
            print(f"  Warning: {score_path} not found, skipping...")
            continue
        
        print(f"  Reading scores from {dataset}...")
        with open(score_path, 'r') as f:
            data = yaml.safe_load(f)
            
            # Filter out None and missing entries
            if data:
                if isinstance(data, dict):
                    if 'missing' not in str(data):
                        merged_scores.append(data)
                elif isinstance(data, list):
                    valid_data = [d for d in data if d is not None and 'missing' not in str(d)]
                    merged_scores.extend(valid_data)
    
    if merged_scores:
        output_scores = save_dir / 'score_uns.yaml'
        with open(output_scores, 'w') as f:
            yaml.dump(merged_scores, f)
        print(f"  Saved combined scores to {output_scores}")
        print(f"  Total score entries: {len(merged_scores)}")
    else:
        print("  No score files found!")
    
    # 3. Combine dataset_uns.yaml files
    print("\n3. Combining dataset_uns.yaml files...")
    merged_datasets = []
    
    for dataset in DATASETS:
        dataset_path = base_dir / dataset / 'dataset_uns.yaml'
        
        if not dataset_path.exists():
            print(f"  Warning: {dataset_path} not found, skipping...")
            continue
        
        print(f"  Reading dataset info from {dataset}...")
        with open(dataset_path, 'r') as f:
            data = yaml.safe_load(f)
            
            if data:
                if isinstance(data, dict):
                    merged_datasets.append(data)
                elif isinstance(data, list):
                    merged_datasets.extend(data)
    
    if merged_datasets:
        output_datasets = save_dir / 'dataset_uns.yaml'
        with open(output_datasets, 'w') as f:
            yaml.dump(merged_datasets, f)
        print(f"  Saved combined datasets to {output_datasets}")
        print(f"  Total dataset entries: {len(merged_datasets)}")
    else:
        print("  No dataset files found!")
    
    # 4. Copy method_configs.yaml and metric_configs.yaml from first dataset
    print("\n4. Copying config files...")
    config_files = ['method_configs.yaml', 'metric_configs.yaml']
    
    for config_file in config_files:
        # Find first dataset that has this file
        for dataset in DATASETS:
            src_path = base_dir / dataset / config_file
            if src_path.exists():
                dst_path = save_dir / config_file
                # Just copy the file directly without parsing (to avoid Groovy YAML issues)
                import shutil
                shutil.copy(src_path, dst_path)
                print(f"  Copied {config_file} from {dataset}")
                break
        else:
            print(f"  Warning: {config_file} not found in any dataset folder")
    
    print("\n✓ All files combined successfully!")
    print(f"Output directory: {save_dir}")

if __name__ == '__main__':
    combine_results()
