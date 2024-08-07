import os
import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt

def calculate_percentile_rank(data, mask_group:list[bool]) -> dict[str, float]:
    '''
        Calculate ranks of the group with the given True values in the mask_group to 1000 random selection of same size by 
        taking sum of top quantile values. 
    '''
    # stats
    n_occurance = (mask_group).sum()
    n_samples = data.shape[0]
    p=[n_occurance/n_samples,  (n_samples-n_occurance)/n_samples]
    # make all expression positive
    X = data
    X += np.abs(np.min(np.min(X)))

    qr_scores = []
    for tt in np.linspace(0.9, .99, 10):
        quantile_per_gene = np.quantile(X, tt, axis=0)
        top_qq_sum = []
        for i_row, row in enumerate(X):
            top_qq_sum.append((row>quantile_per_gene).sum())
        top_qq_sum = np.asarray(top_qq_sum)
              
        # mask the group
        top_qq_ctr_sum = top_qq_sum[mask_group].sum()
        # mask random
        top_qq_random_sum_list = []
        for i in range(1000):
            mask = np.random.choice([True, False], n_samples, p=p)
            top_qq_random_sum_list.append(top_qq_sum[mask].sum())
        score = round(((top_qq_random_sum_list>top_qq_ctr_sum).sum()/1000)*100, 3)        
        qr_scores.append(score)
    
    return np.mean(qr_scores)
def degree_centrality(net, source='source', target='target', normalize=False):
    counts = net.groupby(source)[target].nunique().values
    if normalize:
        total_targets = net[target].nunique()
        counts = counts/total_targets
    return counts

class Connectivity:
    def __init__(self, net, **kwargs):
        self.out_deg = degree_centrality(net, source='source', target='target',  **kwargs)
        self.in_deg = degree_centrality(net, source='target', target='source',  **kwargs)
def calculate_hvgs_stats(geneset, hvgs, pert_genes):
    n_random = 1000
    shared_genes = np.intersect1d(geneset, pert_genes)
    shared_hvgs_n = len(np.intersect1d(hvgs, shared_genes))
    shared_hvgs_ratio = shared_hvgs_n/len(shared_genes)

    # to percentile
    random_ratios = []
    for i in range(n_random):
        random_genes = np.random.choice(pert_genes, shared_hvgs_n)
        random_ratios.append(len(np.intersect1d(hvgs, random_genes))/shared_hvgs_n)
    top_p = ((np.asarray(random_ratios)>shared_hvgs_ratio).sum()/n_random)*100

    return {'n_included_hvgs':shared_hvgs_n, 'ratio':shared_hvgs_ratio, 'percentile_rank':round(top_p,1)} 
def annotate_genes(targets, gene_annotation_df):
    gene_annot = gene_annotation_df[gene_annotation_df.Gene.isin(targets)].Transcript_type.value_counts()
        
    gene_annot = round(gene_annot*100/len(targets),1)
    mask_rest = gene_annot<1
    sum_rest = gene_annot[mask_rest].sum()
    gene_annot = gene_annot[~mask_rest]
    gene_annot.loc['Rest'] = round(sum_rest,1)
    return gene_annot
class Explanatory_analysis:
    '''
    This class provides functions for explanatory analysis of GRNs including topology analysis and biological annotaions.
    '''
    def __init__(self, net, peak_gene_net=None):
        self.net = net 
        self.peak_gene_net = peak_gene_net
        #TODO: check the format. [source, target, weight]

        self.tfs = net.source.unique()
        self.targets = net.target.unique()
        if peak_gene_net is not None:
            self.cres = peak_gene_net.source.unique()
        # check duplicates
        dup_flag = False
        if 'cell_type' in net.columns:
            if net.duplicated(subset=['source','target','cell_type']).any():
                dup_flag = True
        else:
            if net.duplicated(subset=['source','target']).any():
                dup_flag = True
        if dup_flag:
            raise ValueError('The network has duplicated source target combinations.')
        self.peak_annot = None
        self.hvg_stats = None

        self.calculate_basic_stats()

    def calculate_all(self):
        '''Calculate all metrics 
        '''
        self.calculate_basic_stats()
        self.calculate_centrality_stats()
        self.annotate_peaks()
        self.calculate_hvgs_stats()
    def calculate_basic_stats(self):
        self.n_links = self.net.shape[0]
        self.n_source = self.net.source.nunique()
        self.n_target = self.net.target.nunique()
        self.ratio_positive_negative =  (self.net.weight>=0).sum()/(self.net.shape[0])
    def calculate_centrality_stats(self) -> None:
        '''Calculate network centrality metrics such as in and out degree
        '''
        self.tf_gene = Connectivity(self.net, normalize=True)
        if self.peak_gene_net is None:
            self.peak_gene = None
            self.n_cres = None
        else:
            self.peak_gene = Connectivity(self.peak_gene_net, normalize=False)
            self.n_cres = self.peak_gene_net.source.nunique()
    def plot_cdf(self, values, ax=None, title=''):
        plot_cumulative_density(values, ax=ax, title=title)
    def plot_connectivity(self):
        data_list = [self.tf_gene.out_deg, self.tf_gene.in_deg]
        if self.peak_gene is not None:
            data_list += [self.peak_gene.in_deg]
        for data in data_list:
            plot_cumulative_density(data)
    def calculate_hvgs_stats(self, hvgs, pert_genes) -> dict[str, float]:
        ''' Calculates metrics related to the inclusion of HVGs in the network.
        '''
        self.hvg_stats = calculate_hvgs_stats(self.targets, hvgs, pert_genes)
        return self.hvg_stats
    def annotate_peaks(self, annotation_df) -> dict[str, float]:
        '''Annotate peaks with associated regions on genome.
        '''
        peaks = self.format_peak(self.cres)
        annotation_df = annotation_df[annotation_df.peak.isin(peaks)]
        value_counts = annotation_df.annotation.value_counts()
        sum_values = value_counts.sum()
        value_ratio = ((value_counts/sum_values)*100).round(1)

        self.peak_annot = value_ratio.to_dict()
        return self.peak_annot
    @staticmethod
    def format_peak(peaks:list[str]) -> list[str]:
        '''Convert any peak data to chr:start-end
        '''
        import re
        formatted_peaks = []
        for peak in peaks:
            chr_, start, end = re.split(r'[:\-_]', peak)
            peak = f"{chr_}:{start}-{end}"

            formatted_peaks.append(peak)
        return formatted_peaks
    def annotate_genes(self, gene_annotation_df) -> dict[str, float]:
        '''Annotates genes'''
        targets = self.targets
        self.gene_annot = annotate_genes(targets, gene_annotation_df)
        return self.gene_annot.to_dict()

    
    
   