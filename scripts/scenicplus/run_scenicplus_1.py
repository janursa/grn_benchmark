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

scplus_obj = dill.load(open(os.path.join('output', 'scenicplus/scplus_obj.pkl'), 'rb'))

from scenicplus.enhancer_to_gene import get_search_space, calculate_regions_to_genes_relationships
biomart_host = "http://sep2019.archive.ensembl.org/" 
species='hsapiens'
assembly='hg38'
upstream = [1000, 150000]
downstream = [1000, 150000]
get_search_space(scplus_obj,
                     biomart_host = biomart_host,
                     species = species,
                     assembly = assembly, 
                     upstream = upstream,
                     downstream = downstream) 
calculate_regions_to_genes_relationships(scplus_obj, 
                        ray_n_cpu = 10, 
                        _temp_dir = None,
                        importance_scoring_method = 'GBM')
with open(os.path.join(work_dir, 'scenicplus/scplus_obj.pkl'), 'wb') as f:
            dill.dump(scplus_obj, f, protocol = -1)