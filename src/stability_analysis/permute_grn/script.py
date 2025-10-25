import pandas as pd
import anndata as ad
import sys
import numpy as np
import os 
import random
import os
import argparse
env = os.environ

sys.path.insert(0, env['UTILS_DIR'])
sys.path.insert(0, env['METRICS_DIR'])
from util import naming_convention, process_links
from all_metrics.helper import main as main_metrics
from src.stability_analysis.permute_grn.helper import main as main_impute 
from src.params import get_par


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)

args = parser.parse_args()

par = get_par(args.dataset)

par = {
  **par, 
  **{
  'grns_dir': f"{env['RESULTS_DIR']}/{args.dataset}/",
  'write_dir': f"{env['RESULTS_DIR']}/experiment/permute_grn/",
  'degrees': [0, 10, 20, 50, 100], #[0, 10, 20, 50, 100],
  'analysis_types': ['direction', 'weight', "net", "sign"],
  'methods': ['grnboost', 'ppcor', 'pearson_corr', 'portia', 'scenicplus', 'scprint'],
}
}

os.makedirs(par['write_dir'], exist_ok=True)
os.makedirs(f"{par['write_dir']}/tmp/", exist_ok=True)

  
#------ noise types and degrees ------#
if True:
  for noise_type in par['analysis_types']: # run for each noise type (net, sign, weight)
    for degree in par['degrees']: # run for each degree
      for i, method in enumerate(par['methods']): # run for each method
        par['prediction'] = f"{par['grns_dir']}/{naming_convention(args.dataset, method)}"
        if not os.path.exists(par['prediction']):
          print(f"Skipping {par['prediction']} as it does not exist")
          continue
        par['prediction_n'] = f"{par['write_dir']}/tmp/{args.dataset}_{method}.csv"
        par['degree'] = degree
        par['noise_type'] = noise_type
        
        main_impute(par)
        # run regs 
        par['prediction'] = par['prediction_n']
        score = main_metrics(par)
        score.index = [method]
        if i==0:
          df_all = score
        else:
          df_all = pd.concat([df_all, score])
        print(noise_type, degree, df_all)
        df_all.to_csv(f"{par['write_dir']}/{args.dataset}-{noise_type}-{degree}-scores.csv")
