import dill
import scanpy as sc
import os
from scenicplus.scenicplus_class import create_SCENICPLUS_object
import numpy as np
from scenicplus.cistromes import merge_cistromes
from scenicplus.enhancer_to_gene import get_search_space, calculate_regions_to_genes_relationships
from scenicplus.TF_to_gene import calculate_TFs_to_genes_relationships

_temp_dir = '/beegfs/desy/user/nourisaj/'
n_cpu = 40
work_dir = '../../output'
tf_file = f'{work_dir}/scenicplus/utoronto_human_tfs_v_1.01.txt'


adata = sc.read_h5ad(f'{work_dir}/scRNA/adata_rna.h5ad')
cistopic_obj = dill.load(f'{work_dir}/scenicplus/scATAC/cistopic_obj.pkl', 'rb')
menr = dill.load(open(f'{work_dir}/scenicplus/scATAC/motifs/menr.pkl', 'rb'))

scplus_obj = create_SCENICPLUS_object(
    GEX_anndata = adata,
    cisTopic_obj = cistopic_obj,
    menr = menr,
    bc_transform_func = lambda x: f'{x}___all_donors' #function to convert scATAC-seq barcodes to scRNA-seq ones
)
scplus_obj.X_EXP = np.array(scplus_obj.X_EXP.todense())
# cistrome 
merge_cistromes(scplus_obj)
# with open(f'{work_dir}/scenicplus/grn/scplus_obj.pkl', 'wb') as f:
#             dill.dump(scplus_obj, f, protocol = -1)
print('cistrome merged')
# regression: region to gene  
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
                        ray_n_cpu = n_cpu, 
                        _temp_dir = _temp_dir,
                        importance_scoring_method = 'GBM')
print('number of genes in gene space: ',scplus_obj.uns['search_space'].Gene.unique().shape)
print('number of genes in region-to-gene: ', scplus_obj.uns['region_to_gene'].target.unique().shape)

print('region to gene calculated')

# scplus_obj = dill.load(open(os.path.join(work_dir, 'grn/scplus_obj.pkl'), 'rb'))

# regression: tf to gene
calculate_TFs_to_genes_relationships(scplus_obj, 
                    tf_file = tf_file,
                    ray_n_cpu = n_cpu, 
                    method = 'GBM',
                    _temp_dir = _temp_dir,
                    key= 'TF2G_adj')
print('number of TFs in TF2G_adj: ', scplus_obj.uns['TF2G_adj'].TF.unique().shape)
print('number of genes in TF2G_adj: ', scplus_obj.uns['TF2G_adj'].target.unique().shape)

print('tf gene calculated')
# eRegulon formation 
from scenicplus.grn_builder.gsea_approach import *
from scenicplus.utils import *
from scenicplus.eregulon_enrichment import *

default_params = dict(min_target_genes = 10,
        adj_pval_thr = 1,
        min_regions_per_gene = 0,
        quantiles = (0.85, 0.90, 0.95),
        top_n_regionTogenes_per_gene = (5, 10, 15),
        top_n_regionTogenes_per_region = (),
        binarize_using_basc = True,
        rho_dichotomize_tf2g = True,
        rho_dichotomize_r2g = True,
        rho_dichotomize_eregulon = True,
        rho_threshold = 0.05,
        keep_extended_motif_annot = True,
        merge_eRegulons = True, 
        order_regions_to_genes_by = 'importance',
        order_TFs_to_genes_by = 'importance',
        key_added = 'eRegulons',
        cistromes_key = 'Unfiltered',
        disable_tqdm = False,
        ray_n_cpu = n_cpu,
        _temp_dir = _temp_dir)
alt_params = dict(min_target_genes = 0,
        adj_pval_thr = 1,
        min_regions_per_gene = 0,
        quantiles = (0.7, 0.80, 0.90),
        top_n_regionTogenes_per_gene = (10, 15, 25),
        top_n_regionTogenes_per_region = (),
        binarize_using_basc = True,
        rho_dichotomize_tf2g = True,
        rho_dichotomize_r2g = True,
        rho_dichotomize_eregulon = True,
        rho_threshold = 0,
        keep_extended_motif_annot = True,
        merge_eRegulons = True, 
        order_regions_to_genes_by = 'importance',
        order_TFs_to_genes_by = 'importance',
        key_added = 'eRegulons',
        cistromes_key = 'Unfiltered',
        disable_tqdm = False, 
        ray_n_cpu = n_cpu,
        _temp_dir = _temp_dir)

build_grn(scplus_obj, **alt_params)
format_egrns(scplus_obj,
            eregulons_key = 'eRegulons',
            TF2G_key = 'TF2G_adj',
            key_added = 'eRegulon_metadata')
with open(f'{work_dir}/scenicplus/grn/scplus_obj.pkl', 'wb') as f:
    dill.dump(scplus_obj, f, protocol = -1)
print('e regulon finished')
