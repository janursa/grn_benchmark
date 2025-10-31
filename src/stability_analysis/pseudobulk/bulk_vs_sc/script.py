import os
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import sys

import os
env = os.environ
assert 'TASK_GRN_INFERENCE_DIR' in env, "Please set the TASK_GRN_INFERENCE_DIR environment variable."
TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
RESULT_DIR = env['RESULTS_DIR']

meta = {
    "metrics_dir": f'{TASK_GRN_INFERENCE_DIR}/src/metrics',
    "util_dir": f'{TASK_GRN_INFERENCE_DIR}/src/utils'
}
sys.path.append(meta["metrics_dir"])
sys.path.append(meta["util_dir"])

# from src.methods.pearson_corr.script import main as main_inference
from all_metrics.helper import main as main_metrics
# from ws_distance.consensus.helper import main as main_consensus_ws_distance
# from regression.consensus.helper import main as main_consensus_reg2

def def_par(dataset):
    par = {
        'evaluation_data': f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/evaluation_data/{dataset}_bulk.h5ad',
        'tf_all': f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/prior/tf_all.csv',
        'apply_skeleton': False,
        'apply_tf': True,
        'max_n_links': 50000,
        'layer': 'lognorm',
        'apply_tf_methods': True,
        'reg_type': 'ridge',
        'num_workers': 10,
        'ws_distance_background': f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/prior/ws_distance_background_{dataset}.csv',
        'evaluation_data_sc': f'{TASK_GRN_INFERENCE_DIR}/resources/extended_data/{dataset}_train_sc.h5ad',
        'regulators_consensus': f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/prior/regulators_consensus_{dataset}.json',
        'ws_consensus': f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/prior/ws_consensus_{dataset}.csv',
        'ws_background': f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/prior/ws_distance_background_{dataset}.csv',
    }
    for key, val in par.items():
        if type(val) is str and 'resources' in val:
            if not os.path.exists(val):
                print(f"Warning: {key} path {val} does not exist.")
    return par
def prediction_file_name(dataset, data_type):
    return f'{results_dir}/{dataset}.prediction_{data_type}.h5ad'

def infer_grn(par, dataset):
    from util import corr_net
    adata = ad.read_h5ad(par["rna"], backed='r')
    tf_all = np.loadtxt(par["tf_all"], dtype=str)

    if dataset == 'parsebioscience':
        perturbs = sorted(adata.obs['perturbation'].unique())[:10]
    else:
        perturbs = tf_all
    adata = adata[adata.obs['perturbation'].isin(perturbs)]
    print(adata.shape)
    adata = adata.to_memory()

    net = corr_net(adata, tf_all, par)
    net = net.astype(str)
    net = ad.AnnData(
        X=None,
        uns={
            "method_id": 'pearson_corr',
            "dataset_id": adata.uns['dataset_id'],
            "prediction": net[["source", "target", "weight"]]
        }
    )
    return net

if __name__ == '__main__':
    results_dir = f'{RESULT_DIR}/experiment/bulk_vs_sc/'
    os.makedirs(results_dir, exist_ok=True)
    datasets = ['replogle', 'parsebioscience', 'xaira_HEK293T', 'xaira_HCT116'] #'replogle', 'xaira_HEK293T', 'xaira_HCT116' , 'parsebioscience' #RUN per dataset seperately
    RUN_GRN_INFERNCE = False
    RUN_METRICS = True

    metrics_all = []
    for dataset in datasets:
        par = def_par(dataset)
        print('Processing dataset:', dataset, flush=True)
        if RUN_GRN_INFERNCE:
            # - infer GRNs
            for data_type in ['sc', 'bulk']: 
                print(f"Inferring GRNs for {data_type} data...", flush=True)
                if data_type == 'bulk':
                    par['rna'] = f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/inference_data/{dataset}_rna.h5ad'
                else:
                    par['rna'] = f'{TASK_GRN_INFERENCE_DIR}/resources/extended_data/{dataset}_train_sc.h5ad'
                net = infer_grn(par, dataset)

                par['prediction'] = prediction_file_name(dataset, data_type)
                net.write_h5ad(par['prediction'])
        
        # - consensus 
        # par['regulators_consensus'] = f'{results_dir}/regulators_consensus_{dataset}.json'
        # par['ws_consensus'] = f'{results_dir}/ws_consensus_{dataset}.json'
        # if True:
        #     def naming_convention(dataset, model):
        #         return f'{dataset}.{model}.h5ad'
            
        #     par['naming_convention'] = naming_convention
        #     par['dataset'] = dataset
        #     par['models_dir'] = results_dir
        #     par['models'] = ['prediction_bulk', 'prediction_sc']
        #     _ = main_consensus_reg2(par)
        #     main_consensus_ws_distance(par)

        # - grn evaluation
        if RUN_METRICS:
            rr_all_store = []
            for data_type in ['sc', 'bulk']:
                print(f"Calculating metrics for {data_type} data...", flush=True)
                par['prediction'] = prediction_file_name(dataset, data_type)
                rr =  main_metrics(par)
                rr['data_type'] = data_type
                rr_all_store.append(rr)
            rr_all = pd.concat(rr_all_store, axis=0)
            rr_all['dataset'] = dataset
            rr_all.to_csv(f'{results_dir}/metrics_{dataset}.csv', index=False)
            metrics_all.append(rr_all)
    metrics_all = pd.concat(metrics_all, axis=0)
    metrics_all.to_csv(f'{results_dir}/metrics_all.csv', index=False)
        