from dotenv import load_dotenv
load_dotenv(dotenv_path="../env.sh", override=True)
import os
env = os.environ  

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

    par = {
        'rna': f"{env['EVALUATION_DIR']}/{dataset}_rna.h5ad",
        'atac': f"{env['EVALUATION_DIR']}/{dataset}_atac.h5ad",
        'evaluation_data': f"{env['EVALUATION_DIR']}/{dataset}_bulk.h5ad",
        'evaluation_data_sc': f"{env['EXTENDED_DIR']}/{dataset}_train_sc.h5ad",
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
        
    }
    if dataset in ['replogle', 'adamson', 'norman']:
        par['ground_truth'] = f"{env['TASK_GRN_INFERENCE_DIR']}/resources/grn_benchmark/ground_truth/K562.csv"
        
    return par