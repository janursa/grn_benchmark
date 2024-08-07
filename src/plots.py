import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import matplotlib

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


def plot_umap(adata, color='', palette=None, ax=None, X_label='X_umap',
              bbox_to_anchor=None, legend=True, legend_title='', **kwrds):
    latent = adata.obsm[X_label]
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




def plot_scatter(obs, obs_index, xs, ys, x_label='', y_label='', log=True, log_y=False, figsize=(5, 7)):
    """
        Scatter plot to showcase the distribution of given variables across different groups. 
    """
    n_axes = len(obs_index)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    alpha = .6
    size = 4
    for i_index, index in enumerate(obs_index):
        # i = i_index // n_axes
        j = i_index % n_axes
        ax = axes[j]

        index_vars = obs[index]
        
        if (index=='sm_name'):
            # included_vars = train_sm_names
            # included_vars = index_vars.unique()
            mask = (index_vars.isin(controls2))
            ax.scatter(xs[~mask], ys[~mask], label='Rest', alpha=alpha, color='blue', s=size-1)
            mask = (index_vars.isin(controls2))
            ax.scatter(xs[mask], ys[mask], label='Positive control', alpha=alpha, color='cyan', s=size)
            
        else:
            included_vars = index_vars.unique()
            for i, var in enumerate(included_vars):
                label = var
                mask = (index_vars == var)
                ax.scatter(xs[mask], ys[mask], label=var, alpha=alpha, color=colors_cell_type[i], s=size)

        ax.grid(alpha=0.4, linewidth=1, color='grey', linestyle='--')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if log:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        ax.margins(0.05)
        ax.spines[['right', 'top']].set_visible(False)
        # ax.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')
        prop = {'size': 9}
        
        handles = []
        for kk, label in enumerate(included_vars if (index != 'sm_name') else ['Rest', 'Positive control']):
            if index == 'sm_name':
                handles.append(Patch(facecolor=colors_positive_controls[kk], label=label))
            else:
                handles.append(Patch(facecolor=colors_cell_type[kk], label=label))
        
        ax.legend(handles=handles, prop=prop, bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0, frameon=False)
    plt.tight_layout()
    return fig, axes
    
def plot_stratified_scatter(obs, ax, xs, ys, palette, size=4,  x_label='', y_label='', log_x=False, log_y=False, extra_labels=None, bbox_to_anchor=(1,1)):
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




## Common functions 
def plot_stacked_bar_chart(cell_types_in_drops, title='', xticks=None, 
                           xticklabels=None, colors=None, figsize=(25, 4), 
                           ax=None, legend=False, color_map=None):
    """
        Stacked bar plot to showcase the compound based distribution of cell counts. Adopted from AmbrosM. 
    """
    # Add a column of zeros to the left and compute the cumulative sums
    cc = np.hstack([np.zeros((len(cell_types_in_drops), 1)), cell_types_in_drops])
    cc_cs = cc.cumsum(axis=1)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    cell_types = cell_types_in_drops.columns
    for i, cell_type in enumerate(cell_types):
        if color_map is None:
            color=colors_cell_type[i]
        else:
            color=color_map[cell_type]
        ax.bar(np.arange(len(cc_cs)),
               cc_cs[:,i+1] - cc_cs[:,i],
               bottom=cc_cs[:,i],
               label=cell_types[i], color=color)
         
    ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(np.arange(len(cc_cs)))
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=90)
    if colors is not None:
        for ticklabel, color in zip(ax.get_xticklabels(), colors):
            ticklabel.set_color(color)
    if legend: 
        ax.legend()
    color_legend_handles = [
        matplotlib.patches.Patch(facecolor='red', label='-'),
        matplotlib.patches.Patch(facecolor='blue', label='-'),
        matplotlib.patches.Patch(facecolor='green', label='-'),
    ]
    return fig, ax