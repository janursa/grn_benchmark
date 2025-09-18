import os
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import sys
import os
env = os.environ
TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
RESULTS_DIR = env['RESULTS_DIR']

sys.path.append(env["METRICS_DIR"])
sys.path.append(env["UTILS_DIR"])

from util import naming_convention as naming_convention_main
from regression_2.helper import main as main_reg2
from ws_distance.helper import main as main_ws_distance
from ws_distance.consensus.helper import main as main_consensus_ws_distance
from regression_2.consensus.helper import main as main_consensus_reg2

from src.params import get_par

def naming_convention(dataset, model):
    return f'{dataset}.{model}.h5ad'

if __name__ == '__main__':
    dataset = 'op'
    results_dir = f'{RESULTS_DIR}/experiment/global_grns/'
    os.makedirs(results_dir, exist_ok=True)
    global_grn_dir = f'{TASK_GRN_INFERENCE_DIR}/resources/grn_models/global/'

    global_grns = os.listdir(global_grn_dir) #all the files in the global_grn_dir
    inferred_methods = ['pearson_corr', 'grnboost', 'scenicplus']
    all_models = inferred_methods + global_grns

    for name in global_grns:
        net = pd.read_csv(f'{global_grn_dir}/{name}', index_col=0)
        net = ad.AnnData(X=None, uns={"method_id": name, "dataset_id": dataset, "prediction": net[["source", "target", "weight"]]})
        net.write(f'{results_dir}/{naming_convention(dataset, name)}')

    for name in inferred_methods:
        net_file = f"{TASK_GRN_INFERENCE_DIR}/resources/results/{dataset}/{naming_convention_main(dataset, name)}"
        net = ad.read_h5ad(net_file)
        net.write(f'{results_dir}/{naming_convention(dataset, name)}')

    # - consensus 
    par = get_par(dataset)
    par['dataset'] = dataset
    par['models_dir'] = results_dir
    par['models'] = all_models
    
    if False:
        _ = main_consensus_reg2(par)
        main_consensus_ws_distance(par)

    metrics_all = []
    for model in all_models:
        # - grn evaluation
        print(f"Calculating metrics for {model}...", flush=True)
        par['prediction'] = f'{results_dir}/{naming_convention(dataset, model)}'
        rr_store = []
        metric_reg2 = main_reg2(par)
        rr_store.append(metric_reg2)
        if False:
            _, metric_ws = main_ws_distance(par)
            rr_store.append(metric_ws)
        rr = pd.concat(rr_store, axis=1)
        rr['model'] = model
        metrics_all.append(rr)
    metrics_all = pd.concat(metrics_all, axis=0)
    metrics_all.to_csv(f'{results_dir}/metrics_all.csv', index=False)