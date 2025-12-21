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

# Add grn_benchmark src to path to use get_par
GRN_BENCHMARK_DIR = env.get('GRN_BENCHMARK_DIR', os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(f'{GRN_BENCHMARK_DIR}/src')

meta = {
    "metrics_dir": f'{TASK_GRN_INFERENCE_DIR}/src/metrics',
    "util_dir": f'{TASK_GRN_INFERENCE_DIR}/src/utils'
}
sys.path.append(meta["metrics_dir"])
sys.path.append(meta["util_dir"])

# from src.methods.pearson_corr.script import main as main_inference
from all_metrics.helper import main as main_metrics
from params import get_par
# from ws_distance.consensus.helper import main as main_consensus_ws_distance
# from regression.consensus.helper import main as main_consensus_reg2
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
    datasets = ['xaira_HEK293T', 'xaira_HCT116', 'replogle'] #'replogle', 'xaira_HEK293T', 'xaira_HCT116' , 'parsebioscience' #RUN per dataset seperately
    RUN_GRN_INFERNCE = False
    RUN_METRICS = True

    metrics_all = []
    for dataset in datasets:
        par = get_par(dataset)
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
        