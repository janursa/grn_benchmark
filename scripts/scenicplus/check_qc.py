


import pandas as pd

import pickle
import os
import argparse

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments.')
 # Define arguments
    parser.add_argument('--donor_id', type=str, help='Donor')
    # Parse arguments
    args = parser.parse_args()
    donor_id = args.donor_id

    work_dir = '../output/multiomics/'

    metadata_bc_file_path = os.path.join(work_dir, 'scATAC/quality_control/metadata_bc.pkl')
    with open(metadata_bc_file_path, 'rb') as file:
        metadata_bc = pickle.load(file)
        print(f'{metadata_bc[donor_id]}')
    profile_data_dict_file_path = os.path.join(work_dir, 'scATAC/quality_control/profile_data_dict.pkl')
    with open(profile_data_dict_file_path, 'rb') as file:
        profile_data_dict = pickle.load(file)      

    plot_sample_metrics(profile_data_dict,
            insert_size_distribution_xlim=[0,600],
            ncol=5,
            plot=True,
            save= f'{work_dir}/sample_metrics.pdf')                   

    #[min,  #max]
    QC_filters = {            
        'Log_unique_nr_frag': [.1 , None],
        'FRIP':               [0.45, None],
        'TSS_enrichment':     [5   , None],
        'Dupl_rate':          [None, None]
        
    }

    # Return figure to plot together with other metrics, and cells passing filters. Figure will be saved as pdf.
    from pycisTopic.qc import *
    FRIP_NR_FRAG_fig, FRIP_NR_FRAG_filter=plot_barcode_metrics(metadata_bc[donor_id],
                                        var_x='Log_unique_nr_frag',
                                        var_y='FRIP',
                                        min_x=QC_filters['Log_unique_nr_frag'][0],
                                        max_x=QC_filters['Log_unique_nr_frag'][1],
                                        min_y=QC_filters['FRIP'][0],
                                        max_y=QC_filters['FRIP'][1],
                                        return_cells=True,
                                        return_fig=True,
                                        plot=False)
    # Return figure to plot together with other metrics, and cells passing filters
    TSS_NR_FRAG_fig, TSS_NR_FRAG_filter=plot_barcode_metrics(metadata_bc[donor_id],
                                        var_x='Log_unique_nr_frag',
                                        var_y='TSS_enrichment',
                                        min_x=QC_filters['Log_unique_nr_frag'][0],
                                        max_x=QC_filters['Log_unique_nr_frag'][1],
                                        min_y=QC_filters['TSS_enrichment'][0],
                                        max_y=QC_filters['TSS_enrichment'][1],
                                        return_cells=True,
                                        return_fig=True,
                                        plot=False)
    # Return figure to plot together with other metrics, but not returning cells (no filter applied for the duplication rate  per barcode)
    DR_NR_FRAG_fig=plot_barcode_metrics(metadata_bc[donor_id],
                                        var_x='Log_unique_nr_frag',
                                        var_y='Dupl_rate',
                                        min_x=QC_filters['Log_unique_nr_frag'][0],
                                        max_x=QC_filters['Log_unique_nr_frag'][1],
                                        min_y=QC_filters['Dupl_rate'][0],
                                        max_y=QC_filters['Dupl_rate'][1],
                                        return_cells=False,
                                        return_fig=True,
                                        plot=False,
                                        plot_as_hexbin = True)

    # Plot barcode stats in one figure
    fig=plt.figure(figsize=(10,10))
    plt.subplot(1, 3, 1)
    img = fig2img(FRIP_NR_FRAG_fig)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    img = fig2img(TSS_NR_FRAG_fig)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    img = fig2img(DR_NR_FRAG_fig)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    fig.savefig(f'{work_dir}/qc.png', dpi=1000)


    bc_passing_filters = {donor_id:[]}
    bc_passing_filters[donor_id] = list((set(FRIP_NR_FRAG_filter) & set(TSS_NR_FRAG_filter)))
    pickle.dump(bc_passing_filters,
                open(os.path.join(work_dir, 'scATAC/quality_control/bc_passing_filters.pkl'), 'wb'))
    print(f"{len(bc_passing_filters[donor_id])} barcodes passed QC stats")

