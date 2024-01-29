
import pandas as pd
import argparse, pickle
from op_sc_tools.repo.mm_utils import get_annot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments.')
 # Define arguments
    parser.add_argument('--donor_id', type=str, help='Donor')
    # Parse arguments
    args = parser.parse_args()
    donor_id = args.donor_id
    print('donor ', donor_id)

    work_dir = '../output/multiomics/'
    fragments_dict = {donor_id: f'{work_dir}/fragments/{donor_id}.bed.gz'} # which version of fragments

    
    n_frag = 100 #100
    tss_flank_window = 1000 # 1000 
    tss_window = 50 # 50
    tss_minimum_signal_window = 100 # 100
    tss_rolling_window = 10 #10
    
    annot = get_annot()
    from pycisTopic.qc import *
    path_to_regions = {donor_id:os.path.join(work_dir, 'scATAC/consensus_peak_calling/consensus_regions.bed')}

    metadata_bc, profile_data_dict = compute_qc_stats(
                    fragments_dict = fragments_dict,
                    tss_annotation = annot,
                    stats=['barcode_rank_plot', 'duplicate_rate', 'insert_size_distribution', 'profile_tss', 'frip'],
                    label_list = None,
                    path_to_regions = path_to_regions,
                    n_cpu = 10,
                    valid_bc = None,
                    n_frag = n_frag,
                    n_bc = None,
                    tss_flank_window = tss_flank_window,
                    tss_window = tss_window,
                    tss_minimum_signal_window = tss_minimum_signal_window,
                    tss_rolling_window = tss_rolling_window,
                    remove_duplicates = True,
                    _temp_dir = None)
    print('creating')
    if not os.path.exists(os.path.join(work_dir, 'scATAC/quality_control')):
        os.makedirs(os.path.join(work_dir, 'scATAC/quality_control'))

    pickle.dump(metadata_bc,
                open(os.path.join(work_dir, 'scATAC/quality_control/metadata_bc.pkl'), 'wb'))

    pickle.dump(profile_data_dict,
                open(os.path.join(work_dir, 'scATAC/quality_control/profile_data_dict.pkl'), 'wb'))
