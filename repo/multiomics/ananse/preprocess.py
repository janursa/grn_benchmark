import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import anansescanpy as asc
import os
import numpy as np

work_dir = '../../output'
outputdir=f"{work_dir}/infer/ananse/"
os.makedirs(outputdir)

# Fill in the directories where the h5ad RNA and ATAC objects are located
atac_PBMC=sc.read(f'{work_dir}/scATAC/adata_atac.h5ad')
rna_PBMC=sc.read(f'{work_dir}/scRNA/adata_rna.h5ad')

def filter_chr(peaks):
    chrs = [peak.split(':')[0] for peak in peaks]
    mask = np.asarray([True if ('chr' in chr) else False for chr in chrs])
    return mask
    
mask = filter_chr(atac_PBMC.var.index)

atac_PBMC = atac_PBMC[:, mask]

rna_PBMC.obs['cell_type']=rna_PBMC.obs['cell_type'].str.replace(' ', '-').astype('category')
rna_PBMC.obs['cell_type']=rna_PBMC.obs['cell_type'].str.replace('_', '-').astype('category')
atac_PBMC.obs['cell_type']=atac_PBMC.obs['cell_type'].str.replace(' ', '-').astype('category')
atac_PBMC.obs['cell_type']=atac_PBMC.obs['cell_type'].str.replace('_', '-').astype('category')


minimal=25
asc.export_CPM_scANANSE(anndata=rna_PBMC,
min_cells=minimal,
outputdir=outputdir,
cluster_id="cell_type"
)
asc.export_ATAC_scANANSE(anndata=atac_PBMC,
min_cells=minimal,
outputdir=outputdir,
cluster_id="cell_type"
)
asc.config_scANANSE(anndata=rna_PBMC,
min_cells=minimal,
outputdir=outputdir,
cluster_id="cell_type"
)
asc.DEGS_scANANSE(anndata=rna_PBMC,
min_cells=minimal,
outputdir=outputdir,
cluster_id="cell_type"
)
