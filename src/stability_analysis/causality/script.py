import pandas as pd
import anndata as ad
import sys
import numpy as np
import os 
import random
import os
import argparse
env = os.environ

sys.path.insert(0, env['UTILS_DIR'])
sys.path.insert(0, env['METRICS_DIR'])

from util import naming_convention, process_links, corr_net
# from all_metrics.helper import main as main_metrics
from src.params import get_par

def main_metrics(par):
    # from all_metrics.helper import main as main_metrics
    from regression.helper import main as main_reg
    # from replica_consistency.helper import main as main_rc
    # out_rc = main_rc(par)
    _, output = main_reg(par)
    # output = pd.concat([out_reg, out_rc], axis=1)
    return output

dataset_masks = ['ctr', 'pert', 'both']
GRN_INFERENCE = False
EVALUATION = True

rr_dir = f"{env['RESULTS_DIR']}/experiment/causality/"
os.makedirs(rr_dir, exist_ok=True)

def main_inference(par, adata, d_mask):
    tf_all = np.loadtxt(par["tf_all"], dtype=str)
    net = corr_net(adata, tf_all, par)
    net = net.astype(str)
    output = ad.AnnData(
        X=None,
        uns={
            "method_id": 'pearson_corr',
            "dataset_id": dataset,
            "prediction": net[["source", "target", "weight"]]
        }
    )
    return output

scores_store = []
for dataset in ['xaira_HEK293T', 'xaira_HCT116', 'replogle']:
    par = get_par(dataset)
    write_dir = f"{rr_dir}/{dataset}"
    os.makedirs(write_dir, exist_ok=True)
    os.makedirs(f"{write_dir}/tmp/", exist_ok=True)


    if GRN_INFERENCE:
        print("Generating masked datasets...", flush=True)
        print('Loading data...', flush=True)
        adata = ad.read_h5ad(f"{env['RESOURCES_DIR']}/extended_data/{dataset}_train_sc.h5ad", backed='r')
        for d_mask in dataset_masks:
            if d_mask == 'ctr':
                adata_masked = adata[adata.obs['is_control']].to_memory()
            elif d_mask == 'pert':
                adata_masked = adata[~adata.obs['is_control']].to_memory()
            else:
                adata_masked = adata.to_memory()
            print(f"Writing masked dataset: {d_mask}...", flush=True)
            adata_masked.write_h5ad(f"{write_dir}/tmp/{dataset}_train_sc_{d_mask}.h5ad")

            print('Running GRN inference...', flush=True)        
            print("Running Pearson correlation...", flush=True)
            par['apply_tf_methods'] = False
            net = main_inference(par, adata_masked, d_mask)
            print("Running Pearson correlation with TF applied...", flush=True)
            par['apply_tf_methods'] = True
            net_tf_applied = main_inference(par, adata_masked, d_mask)
            print("Writing output networks...", flush=True)
            net.write_h5ad(f"{write_dir}/tmp/{dataset}_{d_mask}_net.h5ad")
            net_tf_applied.write_h5ad(f"{write_dir}/tmp/{dataset}_{d_mask}_net_tf_applied.h5ad")


    if EVALUATION:
        print('Evaluating inferred networks...', flush=True)
        for d_mask in dataset_masks:
            print(f"Dataset mask: {d_mask}", flush=True)
            par['prediction'] = f"{write_dir}/tmp/{dataset}_{d_mask}_net_tf_applied.h5ad"
            print("Evaluating Pearson with TF applied...", flush=True)
            score_tf_applied = main_metrics(par)
            score_tf_applied['method'] = 'pearson_tf_applied'
            score_tf_applied['dataset_mask'] = d_mask
            score_tf_applied['dataset'] = dataset
            scores_store.append(score_tf_applied)

            print("Evaluating Pearson without TF applied...", flush=True)
            par['prediction'] =f"{write_dir}/tmp/{dataset}_{d_mask}_net.h5ad"
            score = main_metrics(par)
            score['method'] = 'pearson'
            score['dataset_mask'] = d_mask
            score['dataset'] = dataset
            scores_store.append(score)
if len(scores_store) > 0:
    scores_df = pd.concat(scores_store)
    print("Writing scores...", flush=True)
    scores_df.to_csv(f"{rr_dir}/scores.csv")
