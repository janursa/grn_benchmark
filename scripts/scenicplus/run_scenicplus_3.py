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


n_cpu = 10
_temp_dir = None
region_ranking = None
gene_ranking = None
calculate_TF_eGRN_correlation = True
calculate_DEGs_DARs = True
variable = ['GEX_celltype']
save_path = f'output/scenicplus/' 

from scenicplus.grn_builder.gsea_approach import *
from scenicplus.utils import *
from scenicplus.eregulon_enrichment import *

scplus_obj = dill.load(open(os.path.join(work_dir, 'scenicplus/scplus_obj.pkl'), 'rb'))

build_grn(scplus_obj,
            min_target_genes = 10,
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
with open(os.path.join(work_dir, 'scenicplus/scplus_obj.pkl'), 'wb') as f:
    dill.dump(scplus_obj, f, protocol = -1)
print('------ saved stage 1 --------- ')
if 'eRegulon_metadata' not in scplus_obj.uns.keys():
    print('Formatting eGRNs')
    format_egrns(scplus_obj,
                    eregulons_key = 'eRegulons',
                    TF2G_key = 'TF2G_adj',
                    key_added = 'eRegulon_metadata')

if 'eRegulon_signatures' not in scplus_obj.uns.keys():
    print('Converting eGRNs to signatures')
    get_eRegulons_as_signatures(scplus_obj,
                                    eRegulon_metadata_key='eRegulon_metadata', 
                                    key_added='eRegulon_signatures')

if 'eRegulon_AUC' not in scplus_obj.uns.keys():
    print('Calculating eGRNs AUC')
    if region_ranking is None:
        print('Calculating region ranking')
        region_ranking = make_rankings(scplus_obj, target='region')
        with open(os.path.join(save_path,'region_ranking.pkl'), 'wb') as f:
            dill.dump(region_ranking, f, protocol = -1)
    print('Calculating eGRNs region based AUC')
    score_eRegulons(scplus_obj,
            ranking = region_ranking,
            eRegulon_signatures_key = 'eRegulon_signatures',
            key_added = 'eRegulon_AUC', 
            enrichment_type= 'region',
            auc_threshold = 0.05,
            normalize = False,
            n_cpu = n_cpu)
    if gene_ranking is None:
        print('Calculating gene ranking')
        gene_ranking = make_rankings(scplus_obj, target='gene')
        with open(os.path.join(save_path,'gene_ranking.pkl'), 'wb') as f:
            dill.dump(gene_ranking, f, protocol = -1)
    print('Calculating eGRNs gene based AUC')
    score_eRegulons(scplus_obj,
            gene_ranking,
            eRegulon_signatures_key = 'eRegulon_signatures',
            key_added = 'eRegulon_AUC', 
            enrichment_type = 'gene',
            auc_threshold = 0.05,
            normalize= False,
            n_cpu = n_cpu)
print('------ saved stage 2 --------- ')
if calculate_TF_eGRN_correlation is True:
    print('Calculating TF-eGRNs AUC correlation')
    for var in variable:
        from scenicplus.cistromes import *
        generate_pseudobulks(scplus_obj, 
                            variable = var,
                            auc_key = 'eRegulon_AUC',
                            signature_key = 'Gene_based',
                            nr_cells = 5,
                            nr_pseudobulks = 100,
                            seed=555)
        generate_pseudobulks(scplus_obj, 
                                    variable = var,
                                    auc_key = 'eRegulon_AUC',
                                    signature_key = 'Region_based',
                                    nr_cells = 5,
                                    nr_pseudobulks = 100,
                                    seed=555)
        TF_cistrome_correlation(scplus_obj,
                        variable = var, 
                        auc_key = 'eRegulon_AUC',
                        signature_key = 'Gene_based',
                        out_key = var+'_eGRN_gene_based')
        TF_cistrome_correlation(scplus_obj,
                                variable = var, 
                                auc_key = 'eRegulon_AUC',
                                signature_key = 'Region_based',
                                out_key = var+'_eGRN_region_based')
                            
if 'eRegulon_AUC_thresholds' not in scplus_obj.uns.keys():
    print('Binarizing eGRNs AUC')
    binarize_AUC(scplus_obj, 
            auc_key='eRegulon_AUC',
            out_key='eRegulon_AUC_thresholds',
            signature_keys=['Gene_based', 'Region_based'],
            n_cpu=n_cpu)
            
if not hasattr(scplus_obj, 'dr_cell'):
    scplus_obj.dr_cell = {}         
           
if 'RSS' not in scplus_obj.uns.keys():
    print('Calculating eRSS')
    from scenicplus.RSS import *
    for var in variable:
        regulon_specificity_scores(scplus_obj, 
                        var,
                        signature_keys=['Gene_based'],
                        out_key_suffix='_gene_based',
                        scale=False)
        regulon_specificity_scores(scplus_obj, 
                        var,
                        signature_keys=['Region_based'],
                        out_key_suffix='_region_based',
                        scale=False)
                        
if calculate_DEGs_DARs is True:
    from scenicplus.diff_features import get_differential_features
    print('Calculating DEGs/DARs')
    for var in variable:
        get_differential_features(scplus_obj, var, use_hvg = True, contrast_type = ['DEGs', 'DARs'])

with open(os.path.join(work_dir, 'scenicplus/scplus_obj.pkl'), 'wb') as f:
    dill.dump(scplus_obj, f, protocol = -1)