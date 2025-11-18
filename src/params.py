from dotenv import load_dotenv
load_dotenv(dotenv_path="../env.sh", override=True)
import os
import sys
env = os.environ  
sys.path.append(env['UTILS_DIR'])

def get_reg2_par(dataset):
    par = {
        'evaluation_data': f"{env['EVALUATION_DIR']}/{dataset}_bulk.h5ad",
        'tf_all': f"{env['PRIOR_DIR']}/tf_all.csv",
        'apply_skeleton': False,
        'apply_tf': True,
        'max_n_links': 50000,
        'layer': 'lognorm',
        'apply_tf_methods': True,
        'reg_type': 'ridge',
        'num_workers': 20,
        'ws_distance_background': f"{env['PRIOR_DIR']}/ws_distance_background_{dataset}.csv",
        'evaluation_data_sc': f"{env['EXTENDED_DIR']}/{dataset}_train_sc.h5ad",
    }
    return par
def get_par(dataset):
    from config import DATASETS_CELLTYPES
    cell_type = DATASETS_CELLTYPES[dataset]

    par = {
        'rna': f"{env['EVALUATION_DIR']}/{dataset}_rna.h5ad",
        'atac': f"{env['EVALUATION_DIR']}/{dataset}_atac.h5ad",
        'evaluation_data': f"{env['EVALUATION_DIR']}/{dataset}_bulk.h5ad",
        'evaluation_data_sc': f"{env['EXTENDED_DIR']}/{dataset}_train_sc.h5ad",
        'evaluation_data_de': f"{env['EXTENDED_DIR']}/{dataset}_de.h5ad",
        'tf_all': f"{env['PRIOR_DIR']}/tf_all.csv",
        'apply_skeleton': False,
        'apply_tf': True,
        'max_n_links': 50000,
        'layer': 'lognorm',
        'apply_tf_methods': True,
        'reg_type': 'ridge',
        'num_workers': 20,
        'ws_distance_background': f"{env['PRIOR_DIR']}/ws_distance_background_{dataset}.csv",
        'regulators_consensus': f"{env['PRIOR_DIR']}/regulators_consensus_{dataset}.json",
        'ws_consensus': f"{env['PRIOR_DIR']}/ws_consensus_{dataset}.csv",
        'evaluation_data_de': f"{env['EVALUATION_DIR']}/{dataset}_de.h5ad",
        'ground_truth_unibind': f"{env['RESOURCES_DIR']}/grn_benchmark/ground_truth/{cell_type}_unibind.csv",
        'ground_truth_chipatlas': f"{env['RESOURCES_DIR']}/grn_benchmark/ground_truth/{cell_type}_chipatlas.csv",
        'ground_truth_remap': f"{env['RESOURCES_DIR']}/grn_benchmark/ground_truth/{cell_type}_remap.csv"

    }
    return par