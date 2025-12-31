import os
import pandas as pd
import numpy as np
import sys
import anndata as ad


from util import naming_convention, process_links

def impute_fun(par):
  net = ad.read_h5ad(par['prediction'])
  prediction = pd.DataFrame(net.uns['prediction'])
  adata = ad.read_h5ad(par['evaluation_data'])
  genes = adata.var_names.tolist()
  
  prediction = process_links(prediction, par={'max_n_links': 50_000})
  degree = par['degree']/100
  type = par['noise_type']
  if type == 'weight': # add noise to weight
    assert 'weight' in prediction.columns 
    print('Add noise to weight')
    std_dev = prediction['weight'].std()
    noise = np.random.normal(loc=0, scale=degree * std_dev, size=prediction['weight'].shape)
    prediction['weight'] += noise
  elif type == 'net': # shuffle source-target matrix
    print('Permute links by reconnecting to entire gene pool')
    # Get all genes from evaluation data
    adata = ad.read_h5ad(par['evaluation_data'])
    all_genes = adata.var_names.tolist()
    
    # Calculate number of edges to permute
    n_edges = len(prediction)
    n_permute = int(n_edges * degree)
    
    # Select random edges to permute
    permute_indices = np.random.choice(prediction.index, size=n_permute, replace=False)
    
    # For selected edges, randomly assign new targets from all genes
    new_targets = np.random.choice(all_genes, size=n_permute, replace=True)
    prediction.loc[permute_indices, 'target'] = new_targets
  elif type == 'sign': # change the regulatory sign
    num_rows = len(prediction)
    num_to_modify = int(num_rows * degree)
    random_indices = np.random.choice(prediction.index, size=num_to_modify, replace=False)
    prediction.loc[random_indices, 'weight'] *= -1
  elif type == 'direction': # change the regulatory sign
    prediction = prediction.reset_index(drop=True)
    n_rows_to_permute = int(len(prediction) * (degree))    
    indices_to_permute = np.random.choice(prediction.index, size=n_rows_to_permute, replace=False)
    prediction.loc[indices_to_permute, ['source', 'target']] = prediction.loc[indices_to_permute, ['target', 'source']].values
  else:
    raise ValueError(f'Wrong type ({type}) for adding noise')
  
  prediction = prediction.astype(str)
  net = ad.AnnData(X=None, uns={"method_id": net.uns['method_id'], "dataset_id": net.uns['dataset_id'], "prediction": prediction[["source", "target", "weight"]]})
  net.write(par['prediction_n'])


def main_metrics(par):
    from all_metrics.helper import main as main_metrics
    # from sem.helper import main as main_sem
    # from rc.helper import main as main_rc
    # from replicate_consistency.helper import main as main_replicate_consistency
    # rr = main_replicate_consistency(par)
    rr = main_metrics(par)
    return rr

def main(par):
  os.makedirs(par['write_dir'], exist_ok=True)
  os.makedirs(f"{par['write_dir']}/tmp/", exist_ok=True)
  #------ noise types and degrees ------#
  for noise_type in par['analysis_types']: # run for each noise type (net, sign, weight)
    for degree in par['degrees']: # run for each degree
      df_all = None
      for i, method in enumerate(par['methods']): # run for each method
        par['prediction'] = f"{par['grns_dir']}/{naming_convention(par['dataset'], method)}"
        if not os.path.exists(par['prediction']):
          print(f"Skipping {par['prediction']} as it does not exist")
          continue
        par['prediction_n'] = f"{par['write_dir']}/tmp/{par['dataset']}_{method}.csv"
        par['degree'] = degree
        par['noise_type'] = noise_type
        
        impute_fun(par)
        # run regs 
        par['prediction'] = par['prediction_n']
        score = main_metrics(par)
        score.index = [method]
        if df_all is None:
          df_all = score
        else:
          df_all = pd.concat([df_all, score])
        print(noise_type, degree, df_all)
      if df_all is not None:
        df_all.to_csv(f"{par['write_dir']}/{par['dataset']}-{noise_type}-{degree}-scores.csv")