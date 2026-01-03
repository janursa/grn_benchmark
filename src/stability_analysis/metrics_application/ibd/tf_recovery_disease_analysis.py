"""
TF Activity Analysis for Disease-Relevant TFs

This script analyzes which TFs become highly active in response to perturbations
(e.g., chemical, bacterial treatments) and identifies disease-relevant TFs.
For datasets like IBD-CD with chemical/bacterial perturbations, we identify TFs
with significant activity changes and connect them to disease pathways.

Usage:
    python tf_recovery_disease_analysis.py --dataset ibd_cd --model pearson_corr
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import anndata as ad
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
from scipy import stats
import subprocess
import tempfile
import warnings
warnings.filterwarnings("ignore")

# Add paths

from grn_benchmark.src.helper import load_env, surrogate_names

env = load_env()
TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
sys.path.append(TASK_GRN_INFERENCE_DIR)

import decoupler as dc

def calculate_tf_activity(de_adata, net_filtered):
    """Calculate TF activity scores for each sample using ULM"""
    print("\nCalculating TF activity scores...")
    
    # Run ULM to get TF activity estimates (new API: dc.mt.ulm modifies adata in place)
    dc.mt.ulm(
        de_adata,
        net_filtered,
        tmin=10
    )
    
    # Extract activity scores and p-values (stored in obsm with key 'score_ulm' and 'padj_ulm')
    if 'score_ulm' in de_adata.obsm:
        tf_activities = de_adata.obsm['score_ulm']  # samples x TFs (DataFrame)
    elif 'ulm_estimate' in de_adata.obsm:
        tf_activities = de_adata.obsm['ulm_estimate']  # samples x TFs (fallback)
    else:
        raise ValueError(f"ULM did not produce activity estimates. Available keys: {list(de_adata.obsm.keys())}")
    
    if 'padj_ulm' in de_adata.obsm:
        tf_pvalues = de_adata.obsm['padj_ulm']  # samples x TFs (DataFrame)
    elif 'ulm_pvals' in de_adata.obsm:
        tf_pvalues = de_adata.obsm['ulm_pvals']  # samples x TFs (fallback)
    else:
        raise ValueError(f"ULM did not produce p-values. Available keys: {list(de_adata.obsm.keys())}")
    
    print(f"TF activity matrix shape: {tf_activities.shape}")
    print(f"Number of TFs evaluated: {tf_activities.shape[1]}")
    
    # Convert to dataframes (already DataFrames from new API)
    tf_activities_df = pd.DataFrame(
        tf_activities.values,
        index=de_adata.obs.index,
        columns=tf_activities.columns
    )
    tf_pvalues_df = pd.DataFrame(
        tf_pvalues.values,
        index=de_adata.obs.index,
        columns=tf_pvalues.columns
    )
    
    # Add perturbation info
    if 'perturbation' in de_adata.obs.columns:
        tf_activities_df['perturbation'] = de_adata.obs['perturbation'].values
        tf_pvalues_df['perturbation'] = de_adata.obs['perturbation'].values
    
    return tf_activities_df, tf_pvalues_df


def analyze_tf_activity(tf_activities_df, tf_pvalues_df, significance_threshold=0.05):
    """Analyze TF activity across all samples
    
    For chemical/bacterial perturbations, we identify TFs that show:
    1. High absolute activity (significantly different from baseline)
    2. Consistency across samples/perturbation types
    """
    print("\nAnalyzing TF activity patterns...")
    
    # Remove perturbation column if exists
    activity_cols = [col for col in tf_activities_df.columns if col != 'perturbation']
    pval_cols = [col for col in tf_pvalues_df.columns if col != 'perturbation']
    
    tf_summary = []
    
    for tf in activity_cols:
        activities = tf_activities_df[tf]
        pvals = tf_pvalues_df[tf]
        
        # Calculate statistics
        mean_activity = activities.mean()
        median_activity = activities.median()
        std_activity = activities.std()
        abs_mean_activity = np.abs(activities).mean()
        
        # Count significant samples
        n_significant = (pvals < significance_threshold).sum()
        n_total = len(activities)
        pct_significant = n_significant / n_total * 100
        
        # Count by direction
        n_upregulated = ((activities > 0) & (pvals < significance_threshold)).sum()
        n_downregulated = ((activities < 0) & (pvals < significance_threshold)).sum()
        
        # Overall significance (mean p-value)
        mean_pval = pvals.mean()
        
        tf_summary.append({
            'TF': tf,
            'mean_activity': mean_activity,
            'median_activity': median_activity,
            'std_activity': std_activity,
            'abs_mean_activity': abs_mean_activity,
            'n_samples': n_total,
            'n_significant': n_significant,
            'pct_significant': pct_significant,
            'n_upregulated': n_upregulated,
            'n_downregulated': n_downregulated,
            'mean_pvalue': mean_pval,
            'is_highly_active': (pct_significant > 50) and (abs_mean_activity > 0.5)
        })
    
    activity_summary_df = pd.DataFrame(tf_summary)
    activity_summary_df = activity_summary_df.sort_values('abs_mean_activity', ascending=False)
    
    print(f"\nTF Activity Summary:")
    print(f"Total TFs evaluated: {len(activity_summary_df)}")
    print(f"TFs active in >50% samples: {(activity_summary_df['pct_significant'] > 50).sum()}")
    print(f"Highly active TFs (>50% samples, |activity| > 0.5): {activity_summary_df['is_highly_active'].sum()}")
    
    return activity_summary_df


def plot_top_de_genes(de_adata, de_results, cell_type, model, output_dir, n_genes=6):
    """
    Plot expression of top DE genes as strip plots with perturbations grouped and diseases side-by-side.
    
    Parameters:
    -----------
    de_adata : AnnData
        AnnData object with expression data and metadata
    de_results : pd.DataFrame
        DataFrame with DE results including p-values and log fold changes
    cell_type : str
        Cell type name
    model : str
        GRN model name
    output_dir : str
        Output directory for plots
    n_genes : int
        Number of top genes to plot
    """
    print(f"\n=== Plotting Top {n_genes} DE Genes ===")
    
    # Get top genes by p-value
    top_genes = de_results.nsmallest(n_genes, 'pvalue_disease')['gene'].values
    
    # Get expression matrix
    if hasattr(de_adata.X, 'toarray'):
        expr_matrix = de_adata.X.toarray()
    else:
        expr_matrix = de_adata.X
    
    # Prepare data for plotting
    obs_df = de_adata.obs.copy()
    obs_df['disease'] = obs_df['disease'].str.upper()  # CD, UC
    
    # Create figure with subplots
    n_cols = 3
    n_rows = int(np.ceil(n_genes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3.5*n_rows))
    axes = axes.flatten() if n_genes > 1 else [axes]
    
    # Color palette for diseases
    disease_colors = {'CD': '#3C5488', 'UC': '#DC0000'}
    
    for idx, gene in enumerate(top_genes):
        ax = axes[idx]
        
        # Get expression for this gene
        gene_idx = np.where(de_adata.var_names == gene)[0][0]
        expr = expr_matrix[:, gene_idx]
        
        # Add expression to dataframe
        plot_df = obs_df.copy()
        plot_df['expression'] = expr
        
        # Get DE stats for title
        gene_stats = de_results[de_results['gene'] == gene].iloc[0]
        logfc = gene_stats['logFC_disease']
        pval = gene_stats['pvalue_disease']
        padj = gene_stats['padj_disease']
        
        # Plot with different perturbations grouped
        perturbations = ['LPS', 'RPMI', 'S. enterica']
        x_positions = {'LPS': 0, 'RPMI': 1, 'S. enterica': 2}
        width = 0.35  # Width for each disease within a perturbation group
        
        for pert in perturbations:
            pert_data = plot_df[plot_df['perturbation'] == pert]
            base_pos = x_positions[pert]
            
            for disease_idx, disease in enumerate(['CD', 'UC']):
                disease_data = pert_data[pert_data['disease'] == disease]['expression']
                
                if len(disease_data) > 0:
                    # Position: center each disease within its perturbation group
                    pos = base_pos + (disease_idx - 0.5) * width
                    
                    # Add jitter to x positions for strip plot
                    x_jitter = np.random.normal(pos, 0.03, size=len(disease_data))
                    
                    # Plot strip
                    ax.scatter(x_jitter, disease_data, 
                              alpha=0.6, s=25, 
                              color=disease_colors[disease],
                              edgecolors='white', linewidths=0.5,
                              label=disease if pert == 'LPS' else '')
                    
                    # Add mean line
                    ax.hlines(disease_data.mean(), pos - 0.1, pos + 0.1,
                             colors=disease_colors[disease], linewidths=2, 
                             alpha=0.9, zorder=10)
        
        # Formatting
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(perturbations, fontsize=10)
        ax.set_ylabel('Expression (log-norm)', fontsize=10)
        ax.set_title(f'{gene}\nlogFC={logfc:.2f}, p={pval:.2e}', 
                    fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlabel('Perturbation', fontsize=10)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(title='Disease', loc='upper right', frameon=True, fontsize=9)
    
    # Remove extra subplots
    for idx in range(n_genes, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_dir}/top_de_genes_{cell_type}_{model}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    plt.close()


def run_limma_de(expr_matrix, design_matrix, gene_names, perturbation_cols):
    """
    Run limma differential expression analysis using R.
    
    Parameters:
    -----------
    expr_matrix : np.ndarray
        Expression matrix (samples x genes)
    design_matrix : pd.DataFrame
        Design matrix with disease and perturbation columns
    gene_names : array-like
        Gene names
    perturbation_cols : list
        List of perturbation column names
    
    Returns:
    --------
    pd.DataFrame : DE results
    """
    print("\n=== Running limma DE analysis via R ===")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_expr:
        expr_file = f_expr.name
        pd.DataFrame(expr_matrix.T, index=gene_names).to_csv(expr_file)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_design:
        design_file = f_design.name
        design_matrix.to_csv(design_file, index=False)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_out:
        output_file = f_out.name
    
    # Create R script
    r_script = f"""
library(limma)

# Read data
expr <- read.csv('{expr_file}', row.names=1)
design <- read.csv('{design_file}')

# Create design matrix
design_mat <- model.matrix(~ disease_encoded + {' + '.join(perturbation_cols)}, data=design)

# Fit limma model
fit <- lmFit(as.matrix(expr), design_mat)
fit <- eBayes(fit)

# Extract results for disease effect
results <- topTable(fit, coef="disease_encoded", number=Inf, sort.by="none")

# Add gene names
results$gene <- rownames(results)
results$logFC_disease <- results$logFC
results$pvalue_disease <- results$P.Value
results$padj_disease <- results$adj.P.Val

# Extract perturbation effects
pert_cols <- c({', '.join([f'"{col}"' for col in perturbation_cols])})
for (col in pert_cols) {{
    if (col %in% colnames(design_mat)) {{
        pert_results <- topTable(fit, coef=col, number=Inf, sort.by="none")
        results[[paste0('coef_pert_', col)]] <- pert_results$logFC
        results[[paste0('pvalue_pert_', col)]] <- pert_results$P.Value
        results[[paste0('padj_pert_', col)]] <- pert_results$adj.P.Val
    }}
}}

# Save results
write.csv(results, '{output_file}', row.names=FALSE)
"""
    
    # Write R script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f_script:
        script_file = f_script.name
        f_script.write(r_script)
    
    try:
        # Run R script
        result = subprocess.run(['Rscript', script_file], 
                               capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"R script error: {result.stderr}")
            raise RuntimeError("limma failed to run")
        
        # Read results
        de_results = pd.read_csv(output_file)
        de_results['converged'] = True
        de_results['failure_reason'] = None
        
        print(f"limma completed successfully")
        
    finally:
        # Clean up temp files
        for f in [expr_file, design_file, output_file, script_file]:
            if os.path.exists(f):
                os.remove(f)
    
    return de_results


def de_analysis(adata, cell_type, test_type='ols'):
    """
    Perform differential expression analysis for one cell type.
    
    Parameters:
    -----------
    adata : AnnData
        Input data
    cell_type : str
        Cell type to analyze
    test_type : str
        Type of test: 'ols' or 'limma'
    
    Fixed effects: disease (CD vs UC), perturbation (LPS vs S. enterica)
    
    Returns AnnData with log fold changes and p-values for disease effect.
    """
    print(f"\n=== DE Analysis for {cell_type} (method: {test_type}) ===")
    
    # Subset to this cell type
    adata_ct = adata[adata.obs['cell_type'] == cell_type].copy()
    
    print(f"Samples: {adata_ct.shape[0]}")
    print(f"Genes (before filtering): {adata_ct.shape[1]}")
    
    # Filter low-expressed genes to reduce noise
    # Keep genes with mean expression > 1 across all samples
    if hasattr(adata_ct.X, 'toarray'):
        expr_matrix = adata_ct.X.toarray()
    else:
        expr_matrix = adata_ct.X
    
    gene_means = expr_matrix.mean(axis=0)
    genes_to_keep = gene_means > 1.0
    adata_ct = adata_ct[:, genes_to_keep].copy()
    
    print(f"Genes (after filtering mean > 1): {adata_ct.shape[1]}")
    print(f"Donors: {adata_ct.obs['donor_id'].nunique()}")
    print(f"Disease distribution: {adata_ct.obs['disease'].value_counts().to_dict()}")
    print(f"Perturbation distribution: {adata_ct.obs['perturbation'].value_counts().to_dict()}")
    
    # Prepare design matrix
    obs_df = adata_ct.obs.copy()
    
    # Check for design issues
    print("\n=== Design Matrix Diagnostics ===")
    
    # Encode categorical variables (case-insensitive)
    obs_df['disease_encoded'] = (obs_df['disease'].str.upper() == 'UC').astype(int)  # UC=1, CD=0
    
    # One-hot encode perturbation (drop first category to avoid collinearity)
    perturbation_dummies = pd.get_dummies(obs_df['perturbation'], prefix='pert', drop_first=True, dtype=int)
    
    # Clean column names for R compatibility (remove spaces, periods, special chars)
    perturbation_dummies.columns = [
        col.replace(' ', '_').replace('.', '_').replace('-', '_') 
        for col in perturbation_dummies.columns
    ]
    
    obs_df = pd.concat([obs_df, perturbation_dummies], axis=1)
    perturbation_cols = perturbation_dummies.columns.tolist()
    
    print(f"Perturbation categories: {obs_df['perturbation'].unique()}")
    print(f"Perturbation dummy columns: {perturbation_cols}")
    
    # Check for perfect separation or collinearity
    design_cols = ['disease_encoded'] + perturbation_cols
    design_matrix = obs_df[design_cols].astype(float).values
    
    # Check rank of design matrix
    rank = np.linalg.matrix_rank(design_matrix)
    expected_rank = len(design_cols)
    
    print(f"Design matrix rank: {rank} (expected: {expected_rank})")
    
    # Check correlation matrix to identify collinearity
    design_df = pd.DataFrame(design_matrix, columns=design_cols)
    corr_matrix = design_df.corr()
    # print(f"\nCorrelation matrix:")
    # print(corr_matrix)
    
    if rank < expected_rank:
        print("\nWARNING: Design matrix is rank deficient - perfect collinearity detected!")
        print("This can happen if:")
        print("  - All samples of one disease have the same perturbation")
        print("  - Not enough variation across groups")
        
        # Check confounding
        print(f"\nSample distribution:")
        print(obs_df.groupby(['disease', 'perturbation']).size())
        
        # Try to proceed with a reduced model (disease only, controlling for perturbation as random effect)
        print("\nProceeding with simplified model: expression ~ disease + (1|donor)")
        print("Note: Perturbation effects cannot be estimated due to collinearity")
        
        # Update design columns to only include disease
        design_cols = ['disease_encoded']
        design_matrix = obs_df[design_cols].astype(float).values
        perturbation_cols = []  # Clear perturbation columns
    
    # Check variance in predictors
    for col in design_cols:
        var = obs_df[col].var()
        if var == 0:
            raise ValueError(f"No variance in predictor {col} - cannot fit model")
        print(f"  {col}: variance = {var:.4f}")
    
    # Get expression matrix
    if hasattr(adata_ct.X, 'toarray'):
        expr_matrix = adata_ct.X.toarray()
    else:
        expr_matrix = adata_ct.X
    
    # Run DE analysis based on test type
    if test_type == 'limma':
        # Use limma via R
        design_df_for_limma = obs_df[['disease_encoded'] + perturbation_cols].copy()
        de_results = run_limma_de(expr_matrix, design_df_for_limma, 
                                   adata_ct.var_names, perturbation_cols)
        n_genes = len(de_results)
        n_zero_variance = 0
        n_failed = 0
        
    else:  # test_type == 'ols'
        # Prepare design matrix with intercept for OLS
        X_design = sm.add_constant(obs_df[design_cols].astype(float))
        
        # Run OLS regression for each gene
        results = []
        n_genes = adata_ct.shape[1]
        n_failed = 0
        n_zero_variance = 0
        
        print(f"\nRunning OLS regression for {n_genes} genes...")
        
        for gene_idx in range(n_genes):
            if gene_idx % 1000 == 0:
                print(f"  Processing gene {gene_idx}/{n_genes}...")
            
            gene_name = adata_ct.var_names[gene_idx]
            y = expr_matrix[:, gene_idx]
            
            # Skip genes with no variance
            if np.var(y) == 0:
                n_zero_variance += 1
                # Initialize perturbation placeholders
                coef_dict = {f'coef_pert_{col}': 0 for col in perturbation_cols}
                pval_dict = {f'pvalue_pert_{col}': 1.0 for col in perturbation_cols}
                
                results.append({
                    'gene': gene_name,
                    'logFC_disease': 0,
                    'pvalue_disease': 1.0,
                    **coef_dict,
                    **pval_dict,
                    'converged': False,
                    'failure_reason': 'zero_variance'
                })
                continue
            
            try:
                # Fit OLS model: expression ~ disease + perturbation
                model = sm.OLS(y, X_design)
                result = model.fit()
                
                # Extract results for disease effect
                logFC_disease = result.params['disease_encoded']
                pval_disease = result.pvalues['disease_encoded']
                
                # Extract results for perturbation effects
                coef_dict = {}
                pval_dict = {}
                for col in perturbation_cols:
                    coef_dict[f'coef_pert_{col}'] = result.params[col]
                    pval_dict[f'pvalue_pert_{col}'] = result.pvalues[col]
                
                results.append({
                    'gene': gene_name,
                    'logFC_disease': logFC_disease,
                    'pvalue_disease': pval_disease,
                    **coef_dict,
                    **pval_dict,
                    'converged': True,
                    'failure_reason': None
                })
            
            except Exception as e:
                n_failed += 1
                coef_dict = {f'coef_pert_{col}': 0 for col in perturbation_cols}
                pval_dict = {f'pvalue_pert_{col}': 1.0 for col in perturbation_cols}
                
                results.append({
                    'gene': gene_name,
                    'logFC_disease': 0,
                    'pvalue_disease': 1.0,
                    **coef_dict,
                    **pval_dict,
                    'converged': False,
                    'failure_reason': str(e)
                })
        
        # Create results dataframe
        de_results = pd.DataFrame(results)
        
        # FDR correction
        from statsmodels.stats.multitest import multipletests
        de_results['padj_disease'] = multipletests(de_results['pvalue_disease'], method='fdr_bh')[1]
        
        # FDR correction for perturbation effects
        for col in perturbation_cols:
            pval_col = f'pvalue_pert_{col}'
            de_results[f'padj_pert_{col}'] = multipletests(de_results[pval_col], method='fdr_bh')[1]
    
    # Summary statistics
    n_converged = de_results['converged'].sum()
    n_sig_disease = (de_results['padj_disease'] < 0.05).sum()
    
    # Additional diagnostics
    min_pval = de_results['pvalue_disease'].min()
    min_padj = de_results['padj_disease'].min()
    n_pval_01 = (de_results['pvalue_disease'] < 0.01).sum()
    n_pval_05 = (de_results['pvalue_disease'] < 0.05).sum()
    
    # Effect sizes
    median_abs_logfc = de_results['logFC_disease'].abs().median()
    max_abs_logfc = de_results['logFC_disease'].abs().max()
    
    print(f"\nResults:")
    print(f"  Genes with zero variance: {n_zero_variance}")
    print(f"  Converged models: {n_converged}/{n_genes}")
    print(f"  Failed to converge: {n_failed}")
    print(f"  Significant genes (disease, FDR<0.05): {n_sig_disease}")
    print(f"\nP-value diagnostics:")
    print(f"  Minimum raw p-value: {min_pval:.4e}")
    print(f"  Minimum adjusted p-value: {min_padj:.4f}")
    print(f"  Genes with p < 0.05 (raw): {n_pval_05}")
    print(f"  Genes with p < 0.01 (raw): {n_pval_01}")
    print(f"\nEffect size diagnostics:")
    print(f"  Median |logFC|: {median_abs_logfc:.4f}")
    print(f"  Maximum |logFC|: {max_abs_logfc:.4f}")
    
    # Show top genes by p-value
    top_genes = de_results.nsmallest(10, 'pvalue_disease')[['gene', 'logFC_disease', 'pvalue_disease', 'padj_disease']]
    print(f"\nTop 10 genes by p-value:")
    print(top_genes.to_string(index=False))
    
    # Create output AnnData
    de_adata = adata_ct.copy()
    
    # Add DE results to var
    de_adata.var['logFC_disease'] = de_results['logFC_disease'].values
    de_adata.var['pvalue_disease'] = de_results['pvalue_disease'].values
    de_adata.var['padj_disease'] = de_results['padj_disease'].values
    
    # Add perturbation results
    for col in perturbation_cols:
        de_adata.var[f'logFC_pert_{col}'] = de_results[f'coef_pert_{col}'].values
        de_adata.var[f'pvalue_pert_{col}'] = de_results[f'pvalue_pert_{col}'].values
        de_adata.var[f'padj_pert_{col}'] = de_results[f'padj_pert_{col}'].values
    
    return de_adata


def main():
    parser = argparse.ArgumentParser(description='Analyze TF activity for disease comparison (CD vs UC)')
    parser.add_argument('--model', type=str, default='pearson_corr',
                        help='GRN model name (e.g., pearson_corr, grnboost)')
    parser.add_argument('--cell_types', type=str, nargs='+',
                        default=['B', 'CD4T', 'CD8T', 'MONO', 'NK'],
                        help='Cell types to analyze')
    parser.add_argument('--test_type', type=str, default='ols', choices=['ols', 'limma'],
                        help='Type of DE test: ols or limma (default: ols)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(env['RESULTS_DIR'], 'experiment', 'tf_disease_comparison')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== TF Activity Analysis: CD vs UC Comparison ===")
    print(f"Model: {args.model}")
    print(f"Cell types: {args.cell_types}")
    print(f"Test type: {args.test_type}")
    print(f"Output directory: {args.output_dir}")
    
    # Load both disease datasets
    print("\n=== Loading Data ===")
    adata_path = f"{TASK_GRN_INFERENCE_DIR}/resources/extended_data/ibd_bulk.h5ad"
    
    print(f"Data path: {adata_path}")
    
    adata = ad.read_h5ad(adata_path)
    adata.X = adata.layers['lognorm']
    
    print(f"\nCombined data shape: {adata.shape}")
    print(f"CD samples: {(adata.obs['disease'] == 'cd').sum()}")
    print(f"UC samples: {(adata.obs['disease'] == 'uc').sum()}")
    
    # Load GRN for the specified model
    # Use CD dataset GRN (assuming same GRN for both diseases)
    prediction_path = f"{TASK_GRN_INFERENCE_DIR}/resources/results/ibd_cd/ibd_cd.{args.model}.{args.model}.prediction.h5ad"
    print(f"\nGRN prediction path: {prediction_path}")
    
    net_adata = ad.read_h5ad(prediction_path)
    net = net_adata.uns['prediction']
    adata = adata[:, adata.var_names.isin(net['target'].unique())].copy()
    
    print(f"Network shape: {net.shape}")
    print(f"Number of TFs in network: {net['source'].nunique()}")
    
    # Filter network to TFs with at least 10 targets
    tf_counts = net['source'].value_counts()
    tfs_to_keep = tf_counts[tf_counts >= 10].index
    net_filtered = net[net['source'].isin(tfs_to_keep)].copy()
    print(f"TFs with >= 10 targets: {len(tfs_to_keep)}")
    
    # Analyze each cell type separately
    all_results = {}
    
    for cell_type in args.cell_types:
        print(f"\n{'='*80}")
        print(f"Processing {cell_type}")
        print(f"{'='*80}")
        
        # Run DE analysis for this cell type
        de_adata = de_analysis(adata, cell_type, test_type=args.test_type)
        
        # Calculate TF activity
        tf_activities_df, tf_pvalues_df = calculate_tf_activity(de_adata, net_filtered)
        
        # Analyze TF activity patterns
        activity_summary_df = analyze_tf_activity(tf_activities_df, tf_pvalues_df)
        
        # Save detailed TF rankings for this cell type
        output_file = os.path.join(args.output_dir, f'tf_activity_summary_{cell_type}_{args.model}.csv')
        activity_summary_df.to_csv(output_file, index=False)
        print(f"\nSaved TF activity summary to: {output_file}")
        
        # Save DE results (gene expression)
        de_output_file = os.path.join(args.output_dir, f'de_results_{cell_type}_{args.model}.csv')
        # Get all disease and perturbation columns
        de_cols = [col for col in de_adata.var.columns if col.startswith(('logFC_', 'pvalue_', 'padj_'))]
        de_results_df = de_adata.var[de_cols].copy()
        de_results_df['gene'] = de_adata.var_names
        de_results_df.to_csv(de_output_file, index=False)
        print(f"Saved DE results to: {de_output_file}")
        
        # Plot top DE genes
        plot_top_de_genes(de_adata, de_results_df, cell_type, args.model, args.output_dir, n_genes=6)
        
        # === NEW: Run DE analysis on TF activities ===
        print(f"\n{'='*80}")
        print(f"Running DE analysis on TF activities")
        print(f"{'='*80}")
        
        # Create AnnData from TF activities
        tf_activity_cols = [col for col in tf_activities_df.columns if col != 'perturbation']
        tf_activity_matrix = tf_activities_df[tf_activity_cols].values
        
        tf_activity_adata = ad.AnnData(
            X=tf_activity_matrix,
            obs=de_adata.obs.copy(),
            var=pd.DataFrame(index=tf_activity_cols)
        )
        tf_activity_adata.var['TF'] = tf_activity_cols
        
        # Run DE analysis on TF activities
        de_tf_adata = de_analysis(
            tf_activity_adata, 
            cell_type, 
            test_type=args.test_type
        )
        
        # Save TF activity DE results
        de_tf_output_file = os.path.join(args.output_dir, f'de_tf_activity_{cell_type}_{args.model}.csv')
        de_tf_cols = [col for col in de_tf_adata.var.columns if col.startswith(('logFC_', 'pvalue_', 'padj_'))]
        de_tf_results_df = de_tf_adata.var[de_tf_cols].copy()
        de_tf_results_df['TF'] = de_tf_adata.var_names
        de_tf_results_df.to_csv(de_tf_output_file, index=False)
        print(f"\nSaved TF activity DE results to: {de_tf_output_file}")
        
        # Print top differential TFs
        print(f"\nTop 10 TFs by differential activity (p-value):")
        top_tfs = de_tf_results_df.nsmallest(10, 'pvalue_disease')[['TF', 'logFC_disease', 'pvalue_disease', 'padj_disease']]
        print(top_tfs.to_string(index=False))
        
        # Analyze by disease groups
        # plot_top_tf_activity_by_perturbation(
        #     activity_summary_df, tf_activities_df, de_adata, cell_type, args.model, args.output_dir
        # )
        
        all_results[cell_type] = {
            'activity_summary': activity_summary_df,
            'de_adata': de_adata,
            'de_tf_adata': de_tf_adata,
            'de_tf_results': de_tf_results_df
        }
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {args.output_dir}")
    
    return all_results


if __name__ == '__main__':
    main()
