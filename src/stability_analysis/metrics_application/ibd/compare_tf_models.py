"""
Compare TF Activity Results Across GRN Models

This script compares TF activity results from different GRN inference methods
to identify:
1. Generic vs disease-specific TFs
2. Overlap/divergence between models
3. Key disease-relevant TFs

Usage:
    python compare_tf_models.py --dataset ibd_cd --model1 grnboost --model2 pearson_corr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
import os
import sys
import argparse

# Add paths
sys.path.append('/Users/jno24/Documents/projs/ongoing/grn_benchmark')
from grn_benchmark.src.helper import load_env

env = load_env()

def load_tf_results(dataset, model, output_dir):
    """Load TF activity summary for a given model"""
    file_path = os.path.join(output_dir, f'tf_activity_summary_{dataset}_{model}.csv')
    df = pd.read_csv(file_path)
    return df

def identify_ibd_tfs():
    """Identify known IBD-relevant TFs"""
    # TFs known to be important in IBD pathogenesis
    ibd_tfs = {
        'NFKB1', 'NFKB2', 'REL', 'RELA', 'RELB',  # NF-kB pathway
        'STAT3', 'STAT4', 'STAT5A', 'STAT5B',      # JAK-STAT pathway
        'SMAD2', 'SMAD3', 'SMAD4', 'SMAD7',        # TGF-beta pathway
        'RORC', 'FOXP3', 'TBX21', 'GATA3',         # T cell differentiation
        'HIF1A', 'HNF4A', 'VDR', 'PPARG',          # Barrier function
        'ATF6', 'XBP1', 'DDIT3',                   # ER stress
        'JUN', 'FOS', 'JUNB', 'JUND',              # AP-1 family
    }
    return ibd_tfs

def compare_models(model1_df, model2_df, model1_name, model2_name, top_n=50):
    """Compare top TFs between two models"""
    print("\n" + "="*80)
    print(f"COMPARISON OF GRN MODELS: {model1_name} vs {model2_name}")
    print("="*80)
    
    # Get top TFs from each model
    model1_top = set(model1_df.nsmallest(top_n, 'mean_pvalue')['TF'].values)
    model2_top = set(model2_df.nsmallest(top_n, 'mean_pvalue')['TF'].values)
    
    # Calculate overlap
    overlap = model1_top & model2_top
    model1_only = model1_top - model2_top
    model2_only = model2_top - model1_top
    
    print(f"\n1. TOP {top_n} TFs OVERLAP:")
    print(f"   - {model1_name}-only TFs: {len(model1_only)}")
    print(f"   - {model2_name}-only TFs: {len(model2_only)}")
    print(f"   - Shared TFs: {len(overlap)}")
    print(f"   - Overlap percentage: {len(overlap)/top_n*100:.1f}%")
    
        
    overlap_generic = overlap

    print(f"   - Generic: {len(overlap_generic)} TFs")
    if overlap_generic:
        print(f"     {sorted(overlap_generic)}")    
    return {
        'overlap': overlap
    }

def analyze_highly_active_tfs(model1_df, model2_df, model1_name, model2_name):
    """Analyze highly active TFs (>50% samples, high activity)"""
    print("\n" + "="*80)
    print("HIGHLY ACTIVE TFs ANALYSIS")
    print("="*80)
    
    model1_active = set(model1_df[model1_df['is_highly_active']]['TF'].values)
    model2_active = set(model2_df[model2_df['is_highly_active']]['TF'].values)
    
    print(f"\nHighly active TFs:")
    print(f"   - {model1_name}: {len(model1_active)} TFs")
    print(f"   - {model2_name}: {len(model2_active)} TFs")
    print(f"   - Shared: {len(model1_active & model2_active)} TFs")
    
    shared_active = model1_active & model2_active
    
    return shared_active

def rank_correlation_analysis(model1_df, model2_df, model1_name, model2_name):
    """Analyze rank correlation between models"""
    print("\n" + "="*80)
    print("RANK CORRELATION ANALYSIS")
    print("="*80)
    
    # Merge datasets on TF
    merged = model1_df[['TF', 'abs_mean_activity', 'pct_significant', 'mean_pvalue']].merge(
        model2_df[['TF', 'abs_mean_activity', 'pct_significant', 'mean_pvalue']],
        on='TF',
        suffixes=(f'_{model1_name}', f'_{model2_name}')
    )
    
    print(f"\nCommon TFs between models: {len(merged)}")
    
    # Calculate correlations
    activity_corr = merged[f'abs_mean_activity_{model1_name}'].corr(merged[f'abs_mean_activity_{model2_name}'])
    pct_corr = merged[f'pct_significant_{model1_name}'].corr(merged[f'pct_significant_{model2_name}'])
    pval_corr = merged[f'mean_pvalue_{model1_name}'].corr(merged[f'mean_pvalue_{model2_name}'])
    
    # print(f"\nPearson correlations:")
    # print(f"   - Absolute activity: {activity_corr:.3f}")
    # print(f"   - % significant samples: {pct_corr:.3f}")
    # print(f"   - Mean p-value: {pval_corr:.3f}")
    
    # Spearman rank correlation
    from scipy.stats import spearmanr
    activity_rank_corr = spearmanr(merged[f'abs_mean_activity_{model1_name}'], 
                                     merged[f'abs_mean_activity_{model2_name}'])[0]
    print(f"   - Activity rank correlation (Spearman): {activity_rank_corr:.3f}")
    
    return merged

def plot_venn_diagram(comparison_results, model1_name, model2_name, output_dir):
    """Plot Venn diagram of TF overlap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    venn = venn2(
        [comparison_results['model1_only'] | comparison_results['overlap'],
         comparison_results['model2_only'] | comparison_results['overlap']],
        set_labels=(model1_name, model2_name),
        ax=ax
    )
    
    # Customize colors
    venn.get_patch_by_id('10').set_color('#ff6b6b')
    venn.get_patch_by_id('01').set_color('#4ecdc4')
    venn.get_patch_by_id('11').set_color('#95e1d3')
    
    plt.title(f'Top 50 TFs Overlap Between GRN Models\n({model1_name} vs {model2_name})', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Save
    output_path = os.path.join(output_dir, f'tf_overlap_venn_{model1_name}_vs_{model2_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved Venn diagram to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare TF activity between two GRN models')
    parser.add_argument('--dataset', type=str, default='ibd_cd',
                        help='Dataset name (e.g., ibd_cd, ibd_uc)')
    parser.add_argument('--model1', type=str, required=True,
                        help='First model name (e.g., grnboost, pearson_corr)')
    parser.add_argument('--model2', type=str, required=True,
                        help='Second model name (e.g., grnboost, pearson_corr)')
    parser.add_argument('--output_dir', type=str, default='output/tf_recovery_disease_analysis',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    dataset = args.dataset
    model1_name = args.model1
    model2_name = args.model2
    output_dir = args.output_dir
    
    print("\n" + "="*80)
    print(f"TF ACTIVITY COMPARISON: {model1_name} vs {model2_name}")
    print(f"Dataset: {dataset.upper()}")
    print("="*80)
    
    # Load results
    model1_df = load_tf_results(dataset, model1_name, output_dir)
    model2_df = load_tf_results(dataset, model2_name, output_dir)
    
    print(f"\nLoaded results:")
    print(f"   - {model1_name}: {len(model1_df)} TFs")
    print(f"   - {model2_name}: {len(model2_df)} TFs")
    
    
    # Compare models
    comparison_results = compare_models(model1_df, model2_df, model1_name, model2_name, top_n=50)
    
    # Analyze highly active TFs
    shared_active = analyze_highly_active_tfs(model1_df, model2_df, model1_name, model2_name)
    
    # Rank correlation
    merged = rank_correlation_analysis(model1_df, model2_df, model1_name, model2_name)
    
    # Plot Venn diagram
    plot_venn_diagram(comparison_results, model1_name, model2_name, output_dir)
    
    print("\n" + "="*80)
    print("SUMMARY OF KEY FINDINGS")
    print("="*80)
    print("\n1. Model Agreement:")
    overlap_pct = len(comparison_results['overlap']) / 50 * 100
    print(f"   - {overlap_pct:.1f}% overlap in top 50 TFs suggests {'high' if overlap_pct > 60 else 'moderate' if overlap_pct > 40 else 'low'} agreement")
    
    print("\n2. Disease Relevance:")
    ibd_ratio = len(comparison_results['overlap_ibd']) / len(comparison_results['overlap']) * 100
    print(f"   - {ibd_ratio:.1f}% of shared TFs are known IBD-relevant")
    
    print("\n3. Generic vs Specific:")
    generic_ratio = len(comparison_results['overlap_generic']) / len(comparison_results['overlap']) * 100
    print(f"   - {generic_ratio:.1f}% of shared TFs are generic/housekeeping")
    print(f"   - Remaining are potentially novel disease-specific TFs")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
