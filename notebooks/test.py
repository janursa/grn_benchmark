
import os
import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tqdm
import json
import warnings
import matplotlib
import sys
import requests
import seaborn as sns
from scipy.stats import spearmanr

# from mplfonts import use_font

sys.path.append('../../')
from grn_benchmark.src.helper import surragate_names
from task_grn_inference.src.utils.util import colors_blind
from task_grn_inference.src.exp_analysis.helper import *


controls3 = ['Dabrafenib', 'Belinostat', 'Dimethyl Sulfoxide']

task_grn_inference_dir = '../../task_grn_inference'
results_folder = '../results_folder/'

plt.rcParams['font.family'] = 'Liberation Sans'
default_font = matplotlib.rcParams['font.family']
print(default_font)

par = {
    'grn_models': ['pearson_corr', 'positive_control', 'celloracle', 'grnboost2', 'scenicplus'],
    'grn_models_dir': f'{task_grn_inference_dir}/resources/grn_models/op/',
    'evaluation_data': f'{task_grn_inference_dir}/resources/evaluation_datasets/op_perturbation.h5ad'
}
# - imports 
sys.path.append('../../task_grn_inference/src/utils')
from task_grn_inference.src.metrics.regression_1.main import main, cross_validation, r2_score, regression_1, process_net

# - read inputs
tf_all = np.loadtxt(f'{task_grn_inference_dir}/resources/prior/tf_all.csv', dtype=str)

perturb_data = ad.read_h5ad(par['evaluation_data'])
perturb_data.X = perturb_data.layers['pearson']
gene_names = perturb_data.var_names
# - calculate the scores and feature importance 
scores_store = []

i_iter = 0
for reg_type in [ "GB", 'ridge']:
    for donor_id in perturb_data.obs.donor_id.unique():
        perturb_data_sub = perturb_data[perturb_data.obs.donor_id == donor_id]
        # perturb_data_sub = perturb_data_sub[:5, :] #TODO: remove this
        obs = perturb_data_sub.obs.reset_index(drop=True)

        for i_model, model in enumerate(par['grn_models']):
            net = pd.read_csv(f"{par['grn_models_dir']}/{model}.csv")
            y_true, y_pred, reg_models = cross_validation(net, perturb_data_sub, par={'exclude_missing_genes':False, 'reg_type':reg_type, 'verbose':3, 'num_workers':20})
            print(model, r2_score(y_true, y_pred, multioutput='variance_weighted'))

            if reg_type == 'ridge':
                coeffs = [reg.coef_ for reg in reg_models]
            else:
                coeffs = [reg.get_feature_importance() for reg in reg_models]

            # - mean of feature importance across CVs
            net_mat = process_net(net.copy(), gene_names)
            mean_coeff = pd.DataFrame(
                np.mean(coeffs, axis=0),
                columns=net_mat.columns,
                index=pd.MultiIndex.from_frame(obs[['sm_name', 'cell_type']])
            )    
            # - normalize feature importance for each sample
            mean_coeff = mean_coeff.abs()
            mean_coeff = mean_coeff.div(mean_coeff.max(axis=1), axis=0)
            
            # - long df for feature importance 
            mean_coeff  = mean_coeff.reset_index()
            mean_coeff = mean_coeff.melt(id_vars=['sm_name', 'cell_type'], var_name='tf', value_name='feature_importance')
            mean_coeff['model'] = model
            mean_coeff['donor_id'] = donor_id
            mean_coeff['reg_type'] = reg_type

            if i_iter == 0:
                feature_importance_all = mean_coeff
            else:
                feature_importance_all = pd.concat([feature_importance_all, mean_coeff], axis=0)

            for i_sample in range(y_true.shape[1]):
                score_sample = r2_score(y_true[:, i_sample], y_pred[:, i_sample], multioutput='variance_weighted')
                scores_store.append({
                    'reg_type': reg_type,
                    'donor_id':donor_id,
                    'r2score':score_sample,
                    'cell_count': obs.loc[i_sample]['cell_count'],
                    'model': model,
                    'cell_type': obs.loc[i_sample]['cell_type'],
                    'sm_name': obs.loc[i_sample]['sm_name']
                })
            i_iter+=1
            
scores_store = pd.DataFrame(scores_store)
scores_store.to_csv('../results_folder/scores_store.csv')
feature_importance_all.to_csv('../results_folder/feature_importance.csv')