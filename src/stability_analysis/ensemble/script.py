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

# grns = ['scenicplus', 'grnboost', 'pearson_corr', 'celloracle', 'scenic']
grns = ['scenicplus', 'grnboost', 'pearson_corr']

TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
sys.path.append(TASK_GRN_INFERENCE_DIR)
from src.utils.util import read_prediction, naming_convention

# Load all predictions
predictions_list = []
for grn in grns:
    par = {
        'prediction': f"{env['RESULTS_DIR']}/{dataset}/{naming_convention(dataset, grn)}"
    }
    prediction = read_prediction(par)
    prediction['weight'] = prediction['weight'].abs()
    
    # Convert to rank PER TARGET GENE
    # Lower rank number = better regulator for that gene
    # rank=1 means this TF is the best regulator for this gene
    prediction['rank_per_gene'] = prediction.groupby('target')['weight'].rank(method='average', ascending=False)
    
    # Normalize to [0,1] within each gene to make methods comparable
    max_rank_gene = prediction.groupby('target')['rank_per_gene'].transform('max')
    prediction['norm_rank'] = prediction['rank_per_gene'] / max_rank_gene
    # norm_rank: 0 to 1, where lower = better regulator for that gene
    
    prediction['method'] = grn
    predictions_list.append(prediction[['source', 'target', 'norm_rank', 'method']])

# Combine all predictions
all_predictions = pd.concat(predictions_list, axis=0)
print(f"Total edges from all methods: {len(all_predictions)}")

# Average normalized ranks for edges that appear in multiple methods
consensus = all_predictions.groupby(['source', 'target'], as_index=False).agg({
    'norm_rank': 'mean',  # Average normalized rank (lower = better)
    'method': 'count'  # Count how many methods predict this edge
})
consensus.columns = ['source', 'target', 'avg_norm_rank', 'n_methods']

# Convert to weight: invert so higher weight = better
# Boost edges that appear in multiple methods
consensus['weight'] = (1 - consensus['avg_norm_rank']) * (1 + 0.3 * (consensus['n_methods'] - 1))

print(f"Unique edges after combining: {len(consensus)}")
print(f"Edges in 1 method: {(consensus['n_methods']==1).sum()}")
print(f"Edges in 2 methods: {(consensus['n_methods']==2).sum()}")
print(f"Edges in 3 methods: {(consensus['n_methods']==3).sum()}")

# Apply network constraints
max_targets_per_tf = 300  # No TF regulates more than 300 genes
max_regulators_per_gene = 100  # No gene regulated by more than 100 TFs
max_edges = 50000  # Total edge limit

print("\nApplying network constraints...")

# Step 1: Limit targets per TF (keep top 300 targets for each TF)
filtered_edges = []
for tf in consensus['source'].unique():
    tf_edges = consensus[consensus['source'] == tf].copy()
    # Keep top max_targets_per_tf by weight
    top_targets = tf_edges.nlargest(min(max_targets_per_tf, len(tf_edges)), 'weight')
    filtered_edges.append(top_targets)

consensus = pd.concat(filtered_edges, axis=0)
print(f"After TF constraint (max {max_targets_per_tf} targets per TF): {len(consensus)} edges")

# Step 2: Limit regulators per gene (keep top 100 regulators for each gene)
filtered_edges = []
for gene in consensus['target'].unique():
    gene_edges = consensus[consensus['target'] == gene].copy()
    # Keep top max_regulators_per_gene by weight
    top_regulators = gene_edges.nlargest(min(max_regulators_per_gene, len(gene_edges)), 'weight')
    filtered_edges.append(top_regulators)

consensus = pd.concat(filtered_edges, axis=0)
print(f"After gene constraint (max {max_regulators_per_gene} regulators per gene): {len(consensus)} edges")

# Step 3: If still over 50k, keep top 50k edges by weight
if len(consensus) > max_edges:
    consensus = consensus.nlargest(max_edges, 'weight')
    print(f"After 50k limit: {len(consensus)} edges")

# Final network
consensus = consensus[['source', 'target', 'weight']].reset_index(drop=True)

print(f"Final consensus network shape: {consensus.shape}")
print(f"Number of target genes covered: {consensus['target'].nunique()}")
print(f"Average regulators per gene: {len(consensus) / consensus['target'].nunique():.1f}")
prediction_c = ad.read_h5ad(f"{env['RESULTS_DIR']}/{dataset}/{naming_convention(dataset, 'grnboost')}")
prediction_c.uns['prediction'] = consensus
prediction_c.uns['method_id'] = 'consensus'
prediction_c.write_h5ad(f"{rr_folder}/{naming_convention(dataset, 'consensus')}")

score = f"{rr_folder}/score_consensus_{dataset}.h5ad"
# run the metrics
cmd = f"cd {env['TASK_GRN_INFERENCE_DIR']} && bash {env['METRICS_DIR']}/all_metrics/run_local.sh --dataset {dataset} --prediction {rr_folder}/{naming_convention(dataset, 'consensus')} --score {score}"
subprocess.run(cmd, shell=True, check=True)
 