
import argparse
import os 
import anndata as ad
import json
import numpy as np
import pandas as pd
import os

from grn_benchmark.src.helper import load_env

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
TEMP_DIR = f"{RESULTS_DIR}/temp/"
import sys
sys.path.append(env['UTILS_DIR'])
from util import naming_convention, process_links
sys.path.append(env['METRICS_DIR'])
from regression_2.helper import cross_validate, net_to_matrix, LabelEncoder, RobustScaler
from ws_distance.helper import main as main_ws_distance

from src.params import get_par as get_base_par

def get_par(dataset):
    par = get_base_par(dataset)
    par = {**par, **{
        'grn_models': ['scenicplus', 'pearson_corr', 'grnboost', 'ppcor'] if dataset=='op' else ['scenic', 'pearson_corr', 'grnboost', 'ppcor'],
        'best_performers': ['pearson_corr', 'scenicplus', 'grnboost'] if dataset=='op' else ['pearson_corr', 'scenic', 'grnboost'],
        'worse_performers': ['ppcor'],
        'grn_models_dir': f'{RESULTS_DIR}/{dataset}/',
        'temp_dir': TEMP_DIR
    }}
    return par

def reg2_func(par):
    perturb_data = ad.read_h5ad(par['evaluation_data'])
    gene_names = perturb_data.var_names
    reg_type = par['reg_type'] 
    if 'donor_id' not in perturb_data.obs:
        perturb_data.obs['donor_id'] = 'donor_one'
        perturb_data.obs['cell_type'] = 'cell_type'
    scores_store = []
    i_iter = 0
    for i_model, model in enumerate(par['grn_models']):
        net = ad.read_h5ad(f"{par['grn_models_dir']}/{naming_convention(dataset, model)}").uns['prediction']
        net = process_links(net, par)
        for donor_id in perturb_data.obs.donor_id.unique():
            perturb_data_sub = perturb_data[perturb_data.obs.donor_id == donor_id]
        
            net_matrix = net_to_matrix(net, gene_names, par)
            n_cells = perturb_data_sub.shape[0]
            random_groups = np.random.choice(range(1, 5+1), size=n_cells, replace=True) # random sampling
            groups = LabelEncoder().fit_transform(random_groups)
            layer = par['layer']
            X = perturb_data_sub.layers[layer]
            try:
                X = X.todense().A
            except:
                pass
            X = RobustScaler().fit_transform(X)
            with open(par['regulators_consensus'], 'r') as f:
                data = json.load(f)
            gene_names_ = np.asarray(list(data.keys()), dtype=object)
            n_features_dict = {gene_name: i for i, gene_name in enumerate(gene_names_)}
            n_features_theta_median = np.asarray([data[gene_name]['0.5'] for gene_name in gene_names], dtype=int)
            tf_names = np.loadtxt(par['tf_all'], dtype=str)
            if par['apply_tf']==False:
                tf_names = gene_names
            rr_all = cross_validate(reg_type, gene_names, tf_names, X, groups, net_matrix, n_features_theta_median, n_jobs=par['num_workers'])
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
    return scores_store


def ws_func(par):
    scores_store = []
    for i_model, model in enumerate(par['grn_models']):
        par['prediction'] = f"{par['grn_models_dir']}/{naming_convention(dataset, model)}"
        rr_fleshed_out, _ = main_ws_distance(par)
        rr_fleshed_out['model'] = model
        scores_store.append(rr_fleshed_out)
    scores_store = pd.concat(scores_store)
    return scores_store

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str)
    args.add_argument('--gene_wise_output', type=str)
    args.add_argument('--ws_output', type=str)
    args = args.parse_args()
    dataset = args.dataset
    gene_wise_output = args.gene_wise_output
    ws_output = args.ws_output
    par = get_par(dataset)
    os.makedirs(par['temp_dir'], exist_ok=True)
    
    if dataset in ['replogle']:
        print('Run ws distance analysis')
        scores_ws = ws_func(par)
        scores_ws['dataset'] = dataset
        scores_ws.to_csv(ws_output)
    print('Run regression 2 gene-wise analysis')
    scores_reg2 = reg2_func(par)
    scores_reg2.to_csv(gene_wise_output)