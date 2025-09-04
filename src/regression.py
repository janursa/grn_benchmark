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
from commons import format_folder

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    lightgbm.LGBMRegressor().set_params(random_state=seed)



class lightgbm_wrapper_simple:
    def __init__(self, params):
        self.params =  params
        
    def fit(self, X_train, Y_train):
        self.n_sample = Y_train.shape[1]
        self.regr_samples = []
        for i in range(self.n_sample):
            regr = lightgbm.LGBMRegressor(**self.params)
            regr.fit(X_train, Y_train[:, i])
            self.regr_samples.append(regr)
            
            
    def predict(self,X_test):
        y_pred_list = []
        for i in range(self.n_sample):
            regr = self.regr_samples[i]
            y_pred = regr.predict(X_test)
            y_pred_list.append(y_pred)
        return np.stack(y_pred_list, axis=1)
from concurrent.futures import ThreadPoolExecutor

class lightgbm_wrapper:
    def __init__(self, params, max_workers=None):
        self.params =  params
        self.max_workers = max_workers
        
    def fit(self, X_train, Y_train):
        self.n_sample = Y_train.shape[1]
        self.regr_samples = [None] * self.n_sample
        
        def fit_sample(i):
            regr = lightgbm.LGBMRegressor(**self.params)
            regr.fit(X_train, Y_train[:, i])
            self.regr_samples[i] = regr
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(fit_sample, range(self.n_sample))
            
    def predict(self, X_test):
        def predict_sample(i):
            regr = self.regr_samples[i]
            return regr.predict(X_test)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            y_pred_list = list(executor.map(predict_sample, range(self.n_sample)))
        
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
    X = StandardScaler().fit_transform(X) #TODO: can go 
    return np.dot(X.T, X) / X.shape[0]

def pivot_grn(net):
    # Remove self-regulations
    net = net[net['source'].astype(str) != net['target'].astype(str)]
    df_tmp = net.pivot(index='target', columns='source', values='weight')
    return df_tmp.fillna(0)
def run_method_1(
            net: pd.DataFrame, 
            train_df: pd.DataFrame,
            reg_type: str = 'GRB',
            exclude_missing_genes: bool = False,
            verbose: int = 0) -> None: 
    """
    net: a df with index as genes and columns as tfs
    train_df: a df with index as genes and columns as samples
    """
    gene_names = train_df.index.to_numpy()
    gene_names_grn = net.index.to_numpy()
    # determine regressor 
    if reg_type=='ridge':
        # regr = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('svc', Ridge(alpha=100, random_state=32))
        # ])
        regr =  Ridge(**dict(random_state=32))
    elif reg_type=='GB':
        regr = lightgbm_wrapper(dict(random_state=32, n_estimators=100, min_samples_leaf=2, min_child_samples=1, feature_fraction=0.05, verbosity=-1))
    elif reg_type=='RF':
        regr = lightgbm_wrapper(dict(boosting_type='rf',random_state=32, n_estimators=100,  feature_fraction=0.05, verbosity=-1))
    
    else:
        raise ValueError("define first")        
    
    n_tfs = net.shape[1]
    # construct feature and target space
    if exclude_missing_genes:
        included_genes = gene_names_grn
    else:
        included_genes = gene_names
    
    X_df = pd.DataFrame(np.zeros((len(included_genes), n_tfs)), index=included_genes)
    Y_df = train_df.loc[included_genes,:]

    mask_shared_genes = X_df.index.isin(net.index)
    print(X_df.shape, Y_df.shape)
    
    # fill the actuall regulatory links
    X_df.loc[mask_shared_genes, :] = net.values
    X = X_df.values.copy()

    # run cv 
    groups = cv_5(len(included_genes))
    # initialize y_pred with the mean of gene expressed across all samples
    means = Y_df.mean(axis=0)
    y_pred = Y_df.copy()
    y_pred[:] = means
    y_pred = y_pred.values

    
    # initialize y_true
    Y = Y_df.values
    y_true = Y.copy()

    unique_groups = np.unique(groups)
    
    for group in unique_groups:
        mask_va = groups == group
        mask_tr = ~mask_va

        # Use logical AND to combine masks correctly
        X_tr = X[mask_tr & mask_shared_genes, :]
        Y_tr = Y[mask_tr & mask_shared_genes, :]

        regr.fit(X_tr, Y_tr)

        y_pred[mask_va & mask_shared_genes, :] = regr.predict(X[mask_va & mask_shared_genes, :])


    # assert ~(y_true==0).any()

    # if verbose >= 1:
    score_r2  = r2_score(y_true, y_pred, multioutput='variance_weighted') #uniform_average', 'variance_weighted
    loss_mse  = mean_squared_error(y_true, y_pred)
    print(f'score_r2: ', score_r2, 'loss_mse: ', loss_mse)


    mean_score_r2 = r2_score(y_true, y_pred, multioutput='variance_weighted')
    gene_scores_r2 = r2_score(y_true.T, y_pred.T, multioutput='raw_values')

    output = dict(mean_score_r2=mean_score_r2, gene_scores_r2=list(gene_scores_r2))

    return output


def main(model_name: str, reg_type: str, norm_method: str, theta: float, tf_n:int, exclude_missing_genes: bool, manipulate: bool, subsample=None, force=False):
    work_dir = '../output'
    DATA_DIR = os.path.join('..', 'output', 'preprocess')
    adata_rna = ad.read_h5ad(os.path.join(DATA_DIR, 'bulk_adata_integrated.h5ad'))
    gene_names = adata_rna.var.index.to_numpy()
    tf_all = np.loadtxt(f'{work_dir}/utoronto_human_tfs_v_1.01.txt', dtype=str)


    train_data = adata_rna.layers[norm_method].copy()
    train_df = pd.DataFrame(train_data, columns=adata_rna.var_names)
    if subsample is not None:
        train_df = train_df.sample(n=subsample, random_state=42) #TODO: remove this
    # check if the file already exist
    print(f'{reg_type=}, {norm_method=}, {model_name=}, {exclude_missing_genes=}')
    folder = format_folder(work_dir, exclude_missing_genes, reg_type, theta, tf_n, norm_method, subsample)
    os.makedirs(folder, exist_ok=True)
    file = f'{folder}/{model_name}_{manipulate}.json'

    if os.path.exists(file):
        if force==False:
            print('Skip running because file exists: ',file)
            return 
    
    # get the grn model
    if model_name in ['positive_control', 'negative_control']:
        net=None 
    else:
        net = pd.read_csv(f'../output/benchmark/grn_models/{model_name}.csv', index_col=0)
    
    # case dependent modifications
    if model_name == 'positive_control':
        net = create_positive_control(train_df)
        net = pd.DataFrame(net, index=gene_names, columns=gene_names)
        net = net.loc[:, net.columns.isin(tf_all)]
        
    elif model_name == 'negative_control':
        ratio = [.98, .01, 0.01]
        net = np.random.choice([0, -1, 1], size=((len(gene_names), 400)),p=ratio)
        net = pd.DataFrame(net, index=gene_names)

    else:
        net = pivot_grn(net)
        net = net[net.index.isin(gene_names)]
        
    # Subset TFs 
    if tf_n is None:
        if False:
            net_sign = net.values
            net_sign[net_sign>0]=1
            net_sign[net_sign<0]=-1
            net_sign = pd.DataFrame(net_sign, index=net.index, columns=net.columns)
            degrees = net_sign.abs().sum(axis=0)
        else:
            degrees = net.abs().sum(axis=0)
        net = net.loc[:, degrees>=degrees.quantile((1-theta))]
    else:

        if tf_n>net.shape[1]:

            print(f'Skip running because tf_n ({tf_n}) is bigger than net.shape[1] ({net.shape[1]})')
            return 
        degrees = net.abs().sum(axis=0)
        net = net[degrees.nlargest(tf_n).index]

    train_df = train_df.T # make it gene*sample

    # manipulate
    if manipulate=='signed':
        net = net.map(lambda x: 1 if x>0 else (-1 if x<0 else 0))


    output = run_method_1(net, train_df, exclude_missing_genes=exclude_missing_genes, reg_type=reg_type)
    
    print(output)
    print(file)
    with open(file, 'w') as f:
        json.dump(output, f)



if __name__ == '__main__':
    set_global_seed(32)

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='theta', help="Type of experiment")
    parser.add_argument('--exclude-missing-genes', action='store_true', help="Exclude missing genes from evaluation.")
    parser.add_argument('--force', action='store_true', help="Force overwrite the files")
    parser.add_argument('--manipulate', type=str, default=None, help="None, signed, shuffle")
    parser.add_argument('--reg_type', type=str, default='ridge', help="Regression type")
    parser.add_argument('--subsample', type=int, default=None, help="Subsample benchamrk data")

    args = parser.parse_args()

    exclude_missing_genes = args.exclude_missing_genes
    experiment = args.experiment
    force = args.force
    manipulate=args.manipulate
    reg_type=args.reg_type
    subsample=args.subsample

    
    print(f'{experiment=}, {exclude_missing_genes=}, {force=}, {manipulate=}, {subsample=}')

    grn_model_names = ['negative_control', 'positive_control'] + ['scglue', 'collectRI', 'figr', 'celloracle', 'granie',  'scenicplus']
    # norm_methods = ['pearson','lognorm','scgen_pearson','scgen_lognorm','seurat_pearson','seurat_lognorm'] #['pearson','lognorm','scgen_pearson','scgen_lognorm','seurat_pearson','seurat_lognorm']
    norm_methods = ['pearson', 'lognorm'] #['pearson','lognorm','scgen_pearson','scgen_lognorm','seurat_pearson','seurat_lognorm']

    if experiment=='default': # default 
        theta = 1.0
        tf_n = None
        for norm_method in norm_methods:
            for grn_model in grn_model_names:
                main(grn_model, reg_type, norm_method, theta, tf_n, exclude_missing_genes, manipulate, subsample, force)
    
    elif experiment=='theta': #experiment with thetas
        thetas = np.linspace(0, 1, 5) # np.linspace(0, 1, 5)
        tf_n = None
        
        for grn_model in grn_model_names:
            for theta in thetas:
                for norm_method in norm_methods:
                    main(grn_model, reg_type, norm_method, theta, tf_n, exclude_missing_genes, manipulate, subsample, force)
    elif experiment=='tf_n':   #experiment with tf_n
        theta = 1.0
        tf_ns = [140]
        for norm_method in norm_methods:
            for grn_model in grn_model_names:
                for tf_n in tf_ns:
                    main(grn_model, reg_type, norm_method, theta, tf_n, exclude_missing_genes, manipulate, subsample,force)
    elif False: #single experiment
        reg_type = 'GB'
        norm_method = 'scgen_pearson'
        # theta = 1
        tf_n = None
        model_name = 'celloracle'
        for theta in [1]:
            main(model_name, reg_type, norm_method, theta, tf_n, exclude_missing_genes, manipulate, force)
    else:
        raise ValueError('define first')
