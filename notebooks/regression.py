import sys
sys.path.append('/lustre1/project/stg_00019/research/Antoine/dependencies')

import os
import argparse
import json
import pandas as pd
import numpy as np
import anndata as ad
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm
import random 


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    lightgbm.LGBMRegressor().set_params(random_state=seed)

set_global_seed(32)

parser = argparse.ArgumentParser()
parser.add_argument('--norm-method', type=str, default='scgen_pearson', help="Normalization method")
parser.add_argument('--reg-type', type=str, default='ridge', help="Regularization type (e.g., 'GB' or 'ridge')")
parser.add_argument('--theta', type=float, default=1, help="Theta value")
parser.add_argument('--manipulate', type=str, default=None, help="None, signed, shuffle")
parser.add_argument('--exclude-missing-genes', action='store_true', help="Exclude missing genes from evaluation.")

args = parser.parse_args()

work_dir = '../output'
DATA_DIR = os.path.join('..', 'output', 'preprocess')
adata_rna = ad.read_h5ad(os.path.join(DATA_DIR, 'bulk_adata_integrated.h5ad'))
gene_names = adata_rna.var.index.to_numpy()
tf_all = np.loadtxt(f'{work_dir}/utoronto_human_tfs_v_1.01.txt', dtype=str)



class lightgbm_wrapper:
    def __init__(self, params):
        self.params =  params
        
    def fit_predict(self, X_train, Y_train, X_test):
        y_pred_list = []
        for i in range(Y_train.shape[1]):
            regr = lightgbm.LGBMRegressor(**self.params)
            regr.fit(X_train, Y_train[:, i])
            y_pred = regr.predict(X_test)
            y_pred_list.append(y_pred)
            
        return np.stack(y_pred_list, axis=1)
def cv_5(genes_n):
    '''5 fold standard'''
    num_groups = 5
    group_size = genes_n // num_groups
    groups = np.repeat(np.arange(num_groups), group_size)
    if genes_n % num_groups != 0:
        groups = np.concatenate((groups, np.arange(genes_n % num_groups)))
    np.random.shuffle(groups)
    return groups

def degree_centrality(net, source='source', target='target', normalize=False):
    counts = net.groupby(source)[target].nunique().values
    if normalize:
        total_targets = net[target].nunique()
        counts = counts / total_targets
    return counts


def create_positive_control(X: np.ndarray) -> np.ndarray:
    X = StandardScaler().fit_transform(X)
    return np.dot(X.T, X) / X.shape[0]


def run_multivariate_gb_regression(
            net: pd.DataFrame, 
            grn_model: str,
            df_train: pd.DataFrame,
            theta: float = 1,
            reg_type: str = 'GB', 
            params: dict = {}, 
            manipulate: bool = False,
            exclude_missing_genes: bool = False,
            verbose: int = 0) -> None:     
    #
    df = df_train.copy()
    df = df.reset_index(level='cell_type').set_index('cell_type') 

    # determine regressor 
    if reg_type=='ridge':
        regr = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', Ridge(alpha=100, random_state=32))
        ])
    elif reg_type=='GB':
        regr = lightgbm_wrapper(params)
    else:
        raise ValueError("define first")

    # for each cell type
    cell_type_index = df.index
    Y_true_matrix_stack = []
    Y_pred_matrix_stack = []

    for cell_type in cell_type_index.unique():
        if verbose >=2:
            print('---- ', cell_type,' --------')
        # subset df for cell type 
        mask = cell_type_index==cell_type
        df_celltype = df[mask]

        if grn_model in ['positive_control', 'negative_control']:
            Y = df_celltype.values.T
            X = create_positive_control(Y.T)
            shared_genes = df.columns
            mask = np.asarray([gene_name in tf_all for gene_name in shared_genes], dtype=bool)
            # print(len(gene_names), len(tf_names), X.shape, Y.shape)
            X = X[:, mask]

            if grn_model == 'negative_control':
                np.random.shuffle(X)

            # Subset TFs based on weights
            degrees = np.mean(np.abs(X), axis=0)
        else:
            # net is cell type dependent or not 
            if 'cell_type' in net:
                net_celltype = net[net.cell_type==cell_type]
            else:
                net_celltype = net.copy()

            # Remove self-regulations
            net_celltype = net_celltype[net_celltype['source'] != net_celltype['target']]

            # match net and df in terms of shared genes 
            net_genes = net_celltype.target.unique()
            shared_genes = np.intersect1d(net_genes, df.columns)
            net_celltype = net_celltype[net_celltype.target.isin(shared_genes)]

            # define X and Y 
            Y = df_celltype[shared_genes].values.T
            df_tmp = net_celltype.pivot(index='target', columns='source', values='weight')
            # tf_names = set(list(df_tmp.columns))
            X = df_tmp.fillna(0).values

            # Subset TFs based on degrees
            degrees = degree_centrality(net_celltype, source='source', target='target', normalize=False)
        mask = (degrees >= np.quantile(degrees, (1-theta)))
        X = X[:, mask]

        if verbose >=2:
            print(f'X (genes, TFs): {X.shape}, Y (genes, samples): {Y.shape}')

        # fill random weights for the missing genes
        if exclude_missing_genes==False:
            # print('filling missing genes with random')
            missing_genes = np.setdiff1d(df.columns, shared_genes)
            tfs_n = X.shape[1]
            sparsity = (X==0).sum()/X.size

            shape = (len(missing_genes), tfs_n)
            if grn_model=='collectRI':
                ratios = [sparsity, (1-sparsity)/2, (1-sparsity)/2]
                
                X_filled = np.random.choice([0, -1, 1], size=shape, p=ratios)
            else:
                X_filled = np.random.normal(np.mean(X), np.std(X), size=shape)
                mask = np.random.choice([0, 1], size=shape, p=[sparsity, 1 - sparsity])
                X_filled = X_filled*mask



            X = np.concatenate([X, X_filled], axis=0)

            Y_missing = df_celltype[missing_genes].values.T
            Y = np.concatenate([Y, Y_missing], axis=0)
            if verbose >=2:
                print(f'X (genes, TFs): {X.shape}, Y (genes, samples): {Y.shape}')

        
        # if manipulate=='shuffled':
        #     X_f = X.flatten()
        #     np.random.shuffle(X_f)
        #     X = X_f.reshape(X.shape)
        if manipulate == 'signed':
            X[X>0]=1
            X[X<0]=-1
            
        # define cv scheme
        groups = cv_5(X.shape[0])

        # run cv 
        Y_pred_stack = []
        Y_true_stack = []
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            mask_va = groups==group
            mask_tr = ~mask_va

            X_tr, Y_tr = X[mask_tr,:], Y[mask_tr,:]
            X_va, Y_true = X[mask_va,:], Y[mask_va,:]

            if reg_type == 'GB':
                Y_pred = regr.fit_predict(X_tr, Y_tr, X_va)
            else:
                regr.fit(X_tr, Y_tr)
                Y_pred = regr.predict(X_va)

            Y_pred_stack.append(Y_pred)
            Y_true_stack.append(Y_true)
        y_pred = np.concatenate(Y_pred_stack, axis=0)
        y_true = np.concatenate(Y_true_stack, axis=0)
        if verbose >= 1:
            score_r2  = r2_score(y_true, y_pred, multioutput='variance_weighted') #uniform_average', 'variance_weighted
            loss_mse  = mean_squared_error(y_true, y_pred)
            print(f'score_r2: ', score_r2, 'loss_mse: ', loss_mse)
        Y_true_matrix_stack.append(y_true)
        Y_pred_matrix_stack.append(y_pred)

    Y_true = np.concatenate(Y_true_matrix_stack, axis=1)
    Y_pred = np.concatenate(Y_pred_matrix_stack, axis=1)

    mean_score_r2 = r2_score(Y_true, Y_pred, multioutput='variance_weighted')
    gene_scores_r2 = r2_score(Y_true.T, Y_pred.T, multioutput='raw_values')


    output = dict(mean_score_r2=mean_score_r2, gene_scores_r2=list(gene_scores_r2))

    return output

def main(grn_model: str, reg_type: str, norm_method: str, theta: float,  exclude_missing_genes: bool, manipulate: bool):

    X = adata_rna.layers[norm_method]

    df_train = {
        'sm_name': adata_rna.obs.sm_name.to_numpy(),
        'cell_type': adata_rna.obs.cell_type.to_numpy(),
        'plate_name': adata_rna.obs.plate_name.to_numpy(),
        'row': adata_rna.obs.row.to_numpy(),
    }
    for j, gene_name in enumerate(gene_names):
        df_train[gene_name] = X[:, j]

    df_train = pd.DataFrame(df_train)
    df_train = df_train.set_index(['sm_name','cell_type','plate_name','row'])

    #df_train = pd.read_csv(f'../resources/df_train/df_train_{norm_method}.csv').set_index(['sm_name','cell_type','plate_name','row'])
    #print(df_train)

    os.makedirs(f'{work_dir}/benchmark/scores/exclude_missing_genes_{exclude_missing_genes}/{reg_type}/{str(theta)}/{norm_method}', exist_ok=True)

    print(grn_model)

    #if os.path.exists(f'{work_dir}/benchmark/scores/{reg_type}/{str(theta)}/{norm_method}/{grn_model}_{manipulate}.json'):
    #    continue

    if grn_model in ['positive_control', 'negative_control']:
        net=None 
    else:

        net = pd.read_csv(f'../output/benchmark/grn_models/{grn_model}.csv', index_col=0)
    

    if reg_type=='ridge':
        output = run_multivariate_gb_regression(net, grn_model, df_train, theta=theta, exclude_missing_genes=exclude_missing_genes, reg_type=reg_type, manipulate=manipulate)
    else:
        output = run_multivariate_gb_regression(net, grn_model, df_train, theta=theta,  exclude_missing_genes=exclude_missing_genes, reg_type=reg_type, 
            params = dict(random_state=32, n_estimators=100, min_samples_leaf=2, min_child_samples=1, feature_fraction=0.05, verbosity=-1))
    print(f'{work_dir}/benchmark/scores/exclude_missing_genes_{exclude_missing_genes}/{reg_type}/{str(theta)}/{norm_method}/{grn_model}_{manipulate}.json')
    with open(f'{work_dir}/benchmark/scores/exclude_missing_genes_{exclude_missing_genes}/{reg_type}/{str(theta)}/{norm_method}/{grn_model}_{manipulate}.json', 'w') as f:
        json.dump(output, f)



reg_type = args.reg_type
norm_method = args.norm_method
theta = args.theta
exclude_missing_genes = args.exclude_missing_genes
manipulate = args.manipulate


grn_model_names = ['collectRI', 'ananse', 'figr', 'celloracle', 'granie', 'scglue', 'scenicplus']
for grn_model in ['positive_control', 'negative_control'] + list(grn_model_names):
# for grn_model in ['scenicplus']:
    for estimator_t in ['ridge']:
        # for norm_method in ['pearson', 'lognorm', 'seurat_pearson', 'seurat_lognorm', 'scgen_pearson', 'scgen_lognorm']:
        # for theta in np.linspace(0, 1, 5):
        #     theta
        for norm_method in ['scgen_pearson']:
            main(grn_model, estimator_t, norm_method, theta, exclude_missing_genes, manipulate)

