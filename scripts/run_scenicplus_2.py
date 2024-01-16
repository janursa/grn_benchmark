import dill
import scanpy as sc
import os
import warnings
warnings.filterwarnings("ignore")
import pandas
import pyranges
# Set stderr to null to avoid strange messages from ray
import sys
_stderr = sys.stderr                                                         
null = open(os.devnull,'wb')
work_dir = 'output'
tmp_dir = None

from scenicplus.TF_to_gene import calculate_TFs_to_genes_relationships
scplus_obj = dill.load(open(os.path.join(work_dir, 'scenicplus/scplus_obj.pkl'), 'rb'))

calculate_TFs_to_genes_relationships(scplus_obj, 
                        tf_file = './output/utoronto_human_tfs_v_1.01.txt',
                        ray_n_cpu = 10, 
                        method = 'GBM',
                        _temp_dir = None,
                        key= 'TF2G_adj')
with open(os.path.join(work_dir, 'scenicplus/scplus_obj.pkl'), 'wb') as f:
        
    dill.dump(scplus_obj, f, protocol = -1)