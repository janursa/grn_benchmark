import os
import sys
import argparse
import subprocess
import pandas as pd
import anndata as ad
import numpy as np

from grn_benchmark.src.helper import load_env

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run ensemble GRN analysis')
parser.add_argument('--rr_folder', type=str, required=True, help='Results folder path')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
args = parser.parse_args()

rr_folder = args.rr_folder
dataset = args.dataset

# Create output directory
os.makedirs(rr_folder, exist_ok=True)

env = load_env()

sys.path.append(env['GRN_BENCHMARK_DIR'])

grns = ['scenicplus', 'grnboost', 'pearson_corr', 'celloracle', 'scenic']

TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
sys.path.append(TASK_GRN_INFERENCE_DIR)
from src.utils.util import read_prediction, naming_convention


prediction_store = []
for grn in grns:
    score = f"{rr_folder}/score_{grn}_{dataset}.h5ad"
    # cmd = f"cd {env['TASK_GRN_INFERENCE_DIR']} && bash {env['METRICS_DIR']}/all_metrics/run_local.sh --dataset {dataset} --prediction {env['RESULTS_DIR']}/{dataset}/{naming_convention(dataset, grn)} --score {score}"
    # subprocess.run(cmd, shell=True, check=True) 
    par = {
        'prediction': f"{env['RESULTS_DIR']}/{dataset}/{naming_convention(dataset, grn)}"
    }
    prediction = read_prediction(par)
    prediction['weight'] = prediction['weight'].abs() 
    prediction['weight'] = (prediction['weight'] - prediction['weight'].mean()) / prediction['weight'].std()

    prediction['method'] = grn
    prediction_store.append(prediction)
prediction = pd.concat(prediction_store, axis=0)

# Strategy 1: Require consensus from multiple methods (vote-based)
# Only keep edges that appear in at least n methods
consensus_counts = prediction.groupby(['source', 'target']).size().reset_index(name='n_methods')
consensus = (
    prediction.groupby(['source', 'target'])['weight']
    .mean()
    .reset_index()
)
consensus = consensus.merge(consensus_counts, on=['source', 'target'])

# Filter: only keep edges supported by at least 2 methods
min_consensus = 2
consensus = consensus[consensus['n_methods'] >= min_consensus].copy()
consensus['weight'] = consensus['weight'] * (consensus['n_methods'] / len(grns))  # Weight by agreement
consensus = consensus[['source', 'target', 'weight']]

print(f"Consensus network shape: {consensus.shape}")
print(f"Original combined edges: {len(prediction)}")
print(f"Edges after requiring {min_consensus}+ methods: {len(consensus)}")
prediction_c = ad.read_h5ad(f"{env['RESULTS_DIR']}/{dataset}/{naming_convention(dataset, 'grnboost')}")
prediction_c.uns['prediction'] = consensus
prediction_c.uns['method_id'] = 'consensus'
prediction_c.write_h5ad(f"{rr_folder}/{naming_convention(dataset, 'consensus')}")

score = f"{rr_folder}/score_consensus_{dataset}.h5ad"
# run the metrics
cmd = f"cd {env['TASK_GRN_INFERENCE_DIR']} && bash {env['METRICS_DIR']}/all_metrics/run_local.sh --dataset {dataset} --prediction {rr_folder}/{naming_convention(dataset, 'consensus')} --score {score}"
subprocess.run(cmd, shell=True, check=True)
 