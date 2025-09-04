


import argparse
import os 
import anndata as ad
import json
import numpy as np
import pandas as pd
from grn_benchmark.src.helper import TASK_GRN_INFERENCE_DIR, RESULT_DIR

meta = {
    'util_dir': f'{TASK_GRN_INFERENCE_DIR}/src/utils/',
    'metrics_dir': f'{TASK_GRN_INFERENCE_DIR}/src/metrics/',
}
import sys
sys.path.append(meta['util_dir'])
from util import naming_convention, process_links
sys.path.append(meta['metrics_dir'])
from regression_2.helper import cross_validate, net_to_matrix, LabelEncoder, RobustScaler


def get_par(dataset):
    par_reg2 = {
        'grn_models': ['scenicplus', 'pearson_corr', 'grnboost', 'ppcor'] if dataset=='op' else ['scenic', 'pearson_corr', 'grnboost', 'ppcor'],
        'best_performers': ['pearson_corr', 'scenicplus', 'grnboost'] if dataset=='op' else ['pearson_corr', 'scenic', 'grnboost'],
        'worse_performers': ['ppcor'],
        'grn_models_dir': f'{TASK_GRN_INFERENCE_DIR}/resources/results/{dataset}/',
        'evaluation_data': f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/evaluation_data/{dataset}_bulk.h5ad',
        'layer': 'lognorm',
        # 'regulators_consensus': f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/prior/regulators_consensus_{dataset}.json',
        'tf_all': f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/prior/tf_all.csv',
        'num_workers': 10,
        'apply_tf': True,
        'apply_skeleton': False,
        'max_n_links': 50_000,
        'temp_dir': f'../output/temp/'
    }
    return par_reg2
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str)
    args.add_argument('--output_file', type=str)

    args = args.parse_args()
    dataset = args.dataset
    output_file = args.output_file
    par_reg2 = get_par(dataset)
    os.makedirs(par_reg2['temp_dir'], exist_ok=True)
    par_reg2['models'] = par_reg2['grn_models']
    par_reg2['regulators_consensus'] = f"{par_reg2['temp_dir']}/regulators_consensus_{dataset}.json"
    par_reg2['models_dir'] = par_reg2['grn_models_dir']
    par_reg2['dataset'] = dataset
    
    print('Compute consensus regulators')
    from regression_2.consensus.helper import main as main_consensus_reg2    
    _ = main_consensus_reg2(par_reg2)

    print('Run regression 2 gene-wise analysis')
    perturb_data = ad.read_h5ad(par_reg2['evaluation_data'])
    gene_names = perturb_data.var_names
    if 'donor_id' not in perturb_data.obs:
        perturb_data.obs['donor_id'] = 'donor_one'
        perturb_data.obs['cell_type'] = 'cell_type'
    scores_store = []
    i_iter = 0
    for reg_type in ['ridge']:
        for donor_id in perturb_data.obs.donor_id.unique():
            perturb_data_sub = perturb_data[perturb_data.obs.donor_id == donor_id]
            obs = perturb_data_sub.obs.reset_index(drop=True)
            for i_model, model in enumerate(par_reg2['grn_models']):
                net = ad.read_h5ad(f"{par_reg2['grn_models_dir']}/{naming_convention(dataset, model)}").uns['prediction']
                net = process_links(net, par_reg2)
                net_matrix = net_to_matrix(net, gene_names, par_reg2)
                n_cells = perturb_data_sub.shape[0]
                random_groups = np.random.choice(range(1, 5+1), size=n_cells, replace=True) # random sampling
                groups = LabelEncoder().fit_transform(random_groups)
                layer = par_reg2['layer']
                X = perturb_data_sub.layers[layer]
                try:
                    X = X.todense().A
                except:
                    pass
                X = RobustScaler().fit_transform(X)
                with open(par_reg2['regulators_consensus'], 'r') as f:
                    data = json.load(f)
                gene_names_ = np.asarray(list(data.keys()), dtype=object)
                n_features_dict = {gene_name: i for i, gene_name in enumerate(gene_names_)}
                n_features_theta_median = np.asarray([data[gene_name]['0.5'] for gene_name in gene_names], dtype=int)
                tf_names = np.loadtxt(par_reg2['tf_all'], dtype=str)
                if par_reg2['apply_tf']==False:
                    tf_names = gene_names
                rr_all = cross_validate(reg_type, gene_names, tf_names, X, groups, net_matrix, n_features_theta_median, n_jobs=par_reg2['num_workers'])
                r2_scores = np.asarray([rr_all['results'][j]['avg-r2'] for j in range(len(rr_all['results']))])
                mean_r2_scores = np.mean(r2_scores)
                for i_gene, gene in enumerate(rr_all['gene_names']):
                    present = gene in net.target.unique()
                    r2score = rr_all['results'][i_gene]['avg-r2']
                    reg_models = rr_all['results'][i_gene]['models']
                    if reg_type == 'ridge':
                        coeffs = [reg.coef_ for reg in reg_models]
                    else:
                        coeffs = [reg.get_feature_importance() for reg in reg_models]
                    coeffs = np.asarray(coeffs)
                    n_regulator = coeffs.shape[1]
                    scores_store.append({
                        'reg_type': reg_type,
                        'donor_id':donor_id,
                        'r2score': r2score,
                        'present':present,
                        'model': model,
                        'gene': gene,
                        'n_regulator': n_regulator,
                        'n_present_regulators': net[net.target==gene]['source'].nunique(),
                        'feature_importance_mean2std': np.mean(np.abs(np.mean(coeffs, axis=0)+1E-6)/(np.std(coeffs, axis=0)+1E-6)).round(3)
                    })
                i_iter+=1
    scores_store = pd.DataFrame(scores_store)
    scores_store.to_csv(output_file)