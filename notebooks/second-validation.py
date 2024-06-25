import sys
sys.path.append('/lustre1/project/stg_00019/research/Antoine/dependencies')

import os
import json
from typing import Dict, List, Tuple, Any, Union
import argparse

import tqdm
import numpy as np
import pandas as pd
import anndata
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm


parser = argparse.ArgumentParser()
parser.add_argument('norm', type=str)
parser.add_argument('estimator', type=str)
parser.add_argument('reduction', type=str)
args = parser.parse_args()
try:
    args.reduction = float(args.reduction)
except ValueError:
    pass


SEED = 0xCAFE

DATA_DIR = os.path.join('..', 'output', 'preprocess')
GRN_DIR = os.path.join('..', 'output', 'benchmark', 'grn_models')
BASELINE_GRN_DIR = os.path.join('..', 'output', 'benchmark', 'baseline_models')
RESULTS_DIR = os.path.join('..', 'output', 'benchmark', 'second-validation')
os.makedirs(RESULTS_DIR, exist_ok=True)

adata_rna = anndata.read_h5ad(os.path.join(DATA_DIR, 'bulk_adata_integrated.h5ad'))

# try: "seurat_lognorm", "scgen_pearson"
adata_rna.layers

groups = LabelEncoder().fit_transform(adata_rna.obs.plate_name)
set(groups)

try:
    gene_names = adata_rna.var.gene.to_numpy()
except:
    gene_names = adata_rna.var.index.to_numpy()
gene_names

norm_t = args.norm
X = adata_rna.layers[norm_t]
X.shape

n_genes = X.shape[1]

X = RobustScaler().fit_transform(X)

def load_grn(filepath: str, gene_names: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    gene_dict = {gene_name: i for i, gene_name in enumerate(gene_names)}
    A = np.zeros((len(gene_names), len(gene_names)), dtype=float)
    df = pd.read_csv(filepath, sep=',', header='infer', index_col=0)
    for source, target, weight in zip(df['source'], df['target'], df['weight']):
        if (source not in gene_dict) or (target not in gene_dict):
            continue
        i = gene_dict[source]
        j = gene_dict[target]
        A[i, j] = float(weight)
    print(f'Sparsity: {np.mean(A == 0)}')
    return A

METHODS = ['collectRI', 'ananse', 'celloracle', 'figr', 'granie', 'scenicplus', 'scglue', 'positive-control', 'negative-control']

grns = []
for method in METHODS:
    if method == 'positive-control':
        #grn = load_grn(os.path.join(BASELINE_GRN_DIR, 'positive_control.csv'), gene_names)
        grn = np.dot(X.T, X) / X.shape[0]
    elif method == 'negative-control':
        grn = 2 * (np.random.rand(n_genes, n_genes) - 0.5)
    else:
        grn = load_grn(os.path.join(GRN_DIR, f'{method}.csv'), gene_names)
    grns.append(grn)


def fill_zeros_in_grn(A: np.ndarray, eps: float = 1e-10):
    A = np.copy(A)
    A[A > 0] = A[A > 0] + eps
    A[A < 0] = A[A < 0] - eps
    A[A == 0] = np.random.rand(*A[A == 0].shape) * 2 * eps - eps
    return A


def consensus_number_of_regulators(reduction_t: Union[str, float], *grns: np.ndarray) -> np.ndarray:
    M = np.asarray(list(grns))
    if isinstance(reduction_t, float):
        return np.ceil(np.quantile(np.sum(M != 0, axis=1), reduction_t, axis=0)).astype(int)
    if reduction_t == 'median':
        return np.ceil(np.median(np.sum(M != 0, axis=1), axis=0)).astype(int)
    elif reduction_t == 'mean':
        return np.ceil(np.mean(np.sum(M != 0, axis=1), axis=0)).astype(int)
    elif reduction_t == 'min':
        return np.ceil(np.min(np.sum(M != 0, axis=1), axis=0)).astype(int)
    elif reduction_t == 'max':
        return np.ceil(np.max(np.sum(M != 0, axis=1), axis=0)).astype(int)
    else:
        raise NotImplementedError(f'Unknown reduction "{reduction_t}"')


reduction_t = args.reduction
n_features = consensus_number_of_regulators(reduction_t, *grns[:-2])

def cross_validate_gene(estimator_t: str, X: np.ndarray, groups: np.ndarray, grn: np.ndarray, j: int, n_features: int = 10) -> Dict[str, float]:
    
    results = {'r2': 0, 'mse': 0}
    
    if n_features == 0:
        return results
    
    # Feature selection
    scores = np.abs(grn[:, j])
    scores[j] = 0
    selected_features = np.argsort(scores)[-n_features:]

    y_pred, y_target, r2s = [], [], []
    for t, (train_index, test_index) in enumerate(LeaveOneGroupOut().split(X, X[:, 0], groups)):

        if estimator_t == 'ridge':
            model = Ridge(random_state=SEED)
        elif estimator_t == 'lasso':
            model = Lasso()
        elif estimator_t == 'linear':
            model = LinearRegression()
        elif estimator_t == 'svm':
            model = LinearSVR(dual='auto', max_iter=10000)
        elif estimator_t == 'gbm':
            model = lightgbm.LGBMRegressor(verbosity=-1, n_estimators=100, n_jobs=4)
        elif estimator_t == 'rf':
            model = RandomForestRegressor()
        elif estimator_t == 'adaboost':
            model = AdaBoostRegressor()
        elif estimator_t == 'knn':
            model = KNeighborsRegressor()
        elif estimator_t == 'mlp':
            model = MLPRegressor(hidden_layer_sizes=(16,))
        else:
            raise NotImplementedError(f'Unknown model "{estimator_t}"')

        X_ = X[:, selected_features]
        y_ = X[:, j]
        X_train = X_[train_index, :]
        X_test = X_[test_index, :]
        y_train = y_[train_index]
        y_test = y_[test_index]
        
        model.fit(X_train, y_train)

        y_pred.append(model.predict(X_test))
        y_target.append(y_test)
        r2s.append(r2_score(y_target[-1], y_pred[-1]))

    y_pred = np.concatenate(y_pred, axis=0)
    y_target = np.concatenate(y_target, axis=0)
    
    results['r2'] = float(r2_score(y_target, y_pred))
    results['avg-r2'] = float(np.mean(r2s))
    results['mse'] = float(mean_squared_error(y_target, y_pred))
    
    return results


def cross_validate(estimator_t: str, gene_names: np.ndarray, X: np.ndarray, groups: np.ndarray, grn: np.ndarray, n_features: np.ndarray) -> List[Dict[str, float]]:
    n_genes = len(grn)
    
    grn = fill_zeros_in_grn(grn)
    
    results = []
    
    for j in tqdm.tqdm(range(n_genes)):
        
        res = cross_validate_gene(estimator_t, X, groups, grn, j, n_features=int(n_features[j]))
        results.append(res)
    
    return {
        'gene_names': list(gene_names),
        'scores': results
    }

estimator_t = args.estimator
override = False

reduction_t = args.reduction
folder = os.path.join(RESULTS_DIR, norm_t, estimator_t, reduction_t if isinstance(reduction_t, str) else str(int(100 * reduction_t)))
os.makedirs(folder, exist_ok=True)
for i, method in enumerate(METHODS):
    if (not override) and os.path.exists(os.path.join(folder, f'{method}.results.json')):
        continue
    print(method, reduction_t, norm_t)
    grn = grns[i]
    results = cross_validate(estimator_t, gene_names, X, groups, grn, n_features)
    with open(os.path.join(folder, f'{method}.results.json'), 'w') as f:
        json.dump(results, f)
