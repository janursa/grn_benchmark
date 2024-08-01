import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
work_dir = '../output'


def format_folder(work_dir, exclude_missing_genes, reg_type, theta, tf_n, norm_method, subsample=None):
    return f'{work_dir}/benchmark/scores/subsample_{subsample}/exclude_missing_genes_{exclude_missing_genes}/{reg_type}/theta_{theta}_tf_n_{tf_n}/{norm_method}'


# COLORS = {
#     'Random': '#74ab8c',
#     'CollectRI': '#83b17b',
#     'FigR': '#96b577',
#     'CellOracle': '#b0b595',
#     'GRaNIE': '#c9b4b1',
#     'ANANSE': '#e2b1cd',
#     'scGLUE': '#e5b8dc',
#     'Scenic+': '#dfc2e5',
#     'HKG': '#e7d2ec',
#     'Positive': 'darkblue',
#     'Negative': 'indianred',
#     'Positive Control': 'darkblue',
#     'Negative Control': 'indianred'
# }

COLORS = {
    'Random': '#74ab8c',
    'CollectRI': '#83b17b',
    'FigR': '#56B4E9',
    'CellOracle': '#b0b595',
    'GRANIE': '#009E73',
    'ANANSE': '#e2b1cd',
    'scGLUE': '#D55E00',
    'Scenic+': '#dfc2e5',
    'HKGs': 'darkblue',
    'Positive': 'darkblue',
    'Negative': 'indianred',
    'Positive Control': 'darkblue',
    'Negative Control': 'indianred'
}

LINESTYLES = {
    'Positive Control': '-',
    'Negative Control': '-',
    'CollectRI': '--',
    'FigR': '-.',
    'CellOracle': ':',
    'GRANIE': ':',
    'ANANSE': '--',
    'scGLUE': '-.',
    'Scenic+': '-',
}

MARKERS = {
    'Random': '.',
    'CollectRI': '.',
    'FigR': 'o',
    'CellOracle': 'd',
    'GRANIE': 'v',
    'ANANSE': 's',
    'scGLUE': 'x',
    'Scenic+': '*',
    'Positive Control': '.',
    'Negative Control': '.'
}

surragate_names = {'CollectRI': 'CollectRI', 'collectRI':'CollectRI', 'collectRI_sign':'CollectRI-signs',
                   'Scenic+': 'Scenic+', 'scenicplus':'Scenic+', 'scenicplus_sign': 'Scenic+-signs',
                   'CellOracle': 'CellOracle', 'celloracle':'CellOracle', 'celloracle_sign':'CellOracle-signs',
                   'figr':'FigR', 'figr_sign':'FigR-signs',
                   'baseline':'Baseline',
                   'cov_net': 'Pearson cov',
                   'granie':'GRANIE',
                   'ananse':'ANANSE',
                   'scglue':'scGLUE',
                   
                   'positive_control':'Positive Control',
                   'negative_control':'Negative Control',
                   'pearson':'APR',
                   'SL':'SLA',
                   'lognorm':'SLA',
                   'seurat_pearson': 'Seurat-APR',
                   'seurat_lognorm': 'Seurat-SLA',
                   'scgen_lognorm': 'scGEN-SLA',
                   'scgen_pearson': 'scGEN-APR',
                   }
controls3 = ['Dabrafenib', 'Belinostat', 'Dimethyl Sulfoxide']
control_compound = 'Dimethyl Sulfoxide'


def shuffle_grn(grn):
    grn_s = grn.copy()
    grn_s['source'] = grn_s['source'].sample(frac=1).reset_index(drop=True)
    grn_s['target'] = grn_s['target'].sample(frac=1).reset_index(drop=True)
    dup_flags = grn_s[['source','target']].duplicated()
    grn_s = grn_s[~dup_flags].reset_index(drop=True)
    if grn_s.duplicated().sum()>0:
        raise ValueError('')
    return grn_s
def sign_grn(grn):
    grn_sign = grn.copy()
    weights = grn_sign.weight
    weights = [1 if weight>0 else -1 for weight in weights]
    grn_sign.weight = weights
    return grn_sign



if False:
    collectRI = pd.read_csv("https://github.com/pablormier/omnipath-static/raw/main/op/collectri-26.09.2023.zip")
    collectRI.to_csv(f'{work_dir}/collectri.csv')




colors_cell_type = ['#c4d9b3', '#c5bc8e', '#c49e81', '#c17d88', 'gray', 'lightsteelblue']
colors_blind = [
          '#E69F00',  # Orange
          '#56B4E9',  # Sky Blue
          '#009E73',  # Bluish Green
          '#F0E442',  # Yellow
          '#0072B2',  # Blue
          '#D55E00',  # Vermillion
          '#CC79A7']  # Reddish Purple

def plot_cumulative_density(data, title='', ax=None, s=1, **kwdgs):
    # Step 1: Sort the data
    sorted_data = np.sort(data)
    
    # Step 2: Compute the cumulative density values
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Step 3: Plot the data
    if ax is None:
    	fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
    	fig = None
    ax.step(sorted_data, cdf, where='post', label=title, **kwdgs)
    ax.set_xlabel('Data')
    ax.set_ylabel('Cumulative Density')
    ax.set_title(title)
    # ax.grid(True)
    return fig, ax
def plot_bar(data_dict: dict[str, np.array], title: str=''):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar(data_dict.keys(), data_dict.values())

    ax.set_xlabel('Methods')
    ax.set_ylabel('Values')
    ax.grid(True, axis='y')  # Add grid only on the y-axis for better readability
    ax.set_title(title)
    aa = plt.xticks(rotation=45)


def plot_umap(adata, color='', palette=None, ax=None, 
              bbox_to_anchor=None, legend=True, legend_title='', **kwrds):
    latent = adata.obsm['X_umap']
    var_unique_sorted = sorted(adata.obs[color].unique())
    legend_handles = []
    
    for i_group, group in enumerate(var_unique_sorted):
        mask = adata.obs[color] == group
        sub_data = latent[mask]
        
        if palette is None:
            c = None 
        else:
            c = palette[i_group]
        
        scatter = ax.scatter(sub_data[:, 0], sub_data[:, 1], label=group, **kwrds, c=c)
        legend_handles.append(plt.Line2D([0], [0], linestyle='none', marker='o', markersize=8, color=scatter.get_facecolor()[0]))
        
        if color == 'leiden':
            mean_x = np.mean(sub_data[:, 0])
            mean_y = np.mean(sub_data[:, 1])
            ax.text(mean_x, mean_y, group, fontsize=9, ha='center', va='top', color='black', weight='bold')

    ax.spines[['right', 'top']].set_visible(False)

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')

    ax.set_xticks([])
    ax.set_yticks([])
    # ax.margins(0.4)
    if legend and color != 'leiden':
        legend  =ax.legend(handles=legend_handles, labels=var_unique_sorted, loc='upper left', 
                  bbox_to_anchor=bbox_to_anchor, frameon=False, title=legend_title, title_fontproperties={'weight': 'bold',  'size':9})
        legend.get_title().set_ha('left')

        # Increase the distance between the title and legend entries
        legend._legend_box.align = "left"  # Align the entries to the left
        # legend._legend_box.set_spacing(2)  # Increase the spacing between the title and entries


def plot_stratified_scatter(obs, ax, xs, ys, size=4, palette=None, x_label='', y_label='', log_x=False, log_y=False, extra_labels=None, bbox_to_anchor=(1,1)):
    """
        Scatter plot to showcase the distribution of a given variable across different groups. 
    """
    alpha = .6
    included_vars = np.unique(obs)
    # included_vars = obs
    for i, sub_var in enumerate(included_vars):        
        mask = (obs == sub_var)
        ax.scatter(xs[mask], ys[mask], label=sub_var, alpha=alpha, color=palette[i], s=size)

    ax.grid(alpha=0.4, linewidth=.5, color='grey', linestyle='--')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.margins(0.05)
    ax.spines[['right', 'top']].set_visible(False)
    # ax.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')
    # prop = {'size': 9}
    prop = {}
    
    handles = []
    for kk, label in enumerate(included_vars):
        if extra_labels is not None:
            extra = extra_labels[label]
            label = f'{label} ({extra})'
        else:
            label = label 
        handles.append(Patch(facecolor=palette[kk], label=label))
    ax.legend(handles=handles, prop=prop, bbox_to_anchor=bbox_to_anchor, loc='upper left', borderaxespad=0, frameon=False)


    for spine in ax.spines.values():
        spine.set_linewidth(0.5)  # Adjust the border thickness here

    # Customize grid lines
    ax.tick_params(axis='both', which='both', width=0.5, length=2)  # Adjust the tick thickness here


def calculate_percentile_rank(bulk_adata_c, mask_group:list[bool]) -> dict[str, float]:
    '''
        Calculate ranks of the group with the given True values in the mask_group to 1000 random selection of same size by 
        taking sum of top quantile values. 
    '''
    # stats
    n_occurance = (mask_group).sum()
    n_samples = len(bulk_adata_c)
    p=[n_occurance/n_samples,  (n_samples-n_occurance)/n_samples]
    # make all expression positive
    X = bulk_adata_c.X
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