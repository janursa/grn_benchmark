
from op_sc_tools.repo.mm_utils import get_chrom_size
def pseudobulk(cell_data, chromsizes, fragments_dict):
    from pycisTopic.pseudobulk_peak_calling import export_pseudobulk
    import os

    bw_paths, bed_paths = export_pseudobulk(input_data = cell_data,
                    variable = 'cell_type',                                                                     # variable by which to generate pseubulk profiles, in this case we want pseudobulks per celltype
                    sample_id_col = 'donor_id', # 
                    chromsizes = chromsizes,
                    bed_path = f'{work_dir}/scATAC/consensus_peak_calling/pseudobulk_bed_files/',  # specify where pseudobulk_bed_files should be stored
                    bigwig_path = f'{work_dir}/scATAC/consensus_peak_calling/pseudobulk_bw_files/',# specify where pseudobulk_bw_files should be stored
                    path_to_fragments = fragments_dict,                                                        # location of fragment fiels
                    n_cpu = 10,                                                                                 # specify the number of cores to use, we use ray for multi processing
                    normalize_bigwig = True,
                    remove_duplicates = True,
                    _temp_dir = None,
                    split_pattern = '-',
                    use_polars=False)
    # del bed_paths['NKcells']
    # del bed_paths['Tregulatorycells']
    # del bed_paths['TcellsCD8_']
    import pickle
    pickle.dump(bed_paths, 
                open(os.path.join(work_dir, 'scATAC/consensus_peak_calling/bed_paths.pkl'), 'wb'))
    pickle.dump(bw_paths,
            open(os.path.join(work_dir, 'scATAC/consensus_peak_calling/bw_paths.pkl'), 'wb'))
def peak_calling():
    # peak calling using MACS2
    # it aligns the reads, calculates pile ups, and detemines summit
    # it calculates enrichment scores compared to a guassian model in the background

    import pickle, os
    from pycisTopic.pseudobulk_peak_calling import peak_calling

    bed_paths = pickle.load(open(os.path.join(work_dir, 'scATAC/consensus_peak_calling/bed_paths.pkl'), 'rb'))
    bw_paths =  pickle.load(open(os.path.join(work_dir, 'scATAC/consensus_peak_calling/bw_paths.pkl'), 'rb'))
    macs_path='macs2'
    # Run peak calling
    narrow_peaks_dict = peak_calling(macs_path,
                                    bed_paths,
                                    os.path.join(work_dir, 'scATAC/consensus_peak_calling/MACS/'),
                                    genome_size='hs',
                                    n_cpu=10,
                                    input_format='BEDPE',
                                    shift=shift, 
                                    ext_size=ext_size,
                                    keep_dup = 'all',
                                    q_value = 0.05,
                                    _temp_dir = None)
    pickle.dump(narrow_peaks_dict, 
                open(os.path.join(work_dir, 'scATAC/consensus_peak_calling/MACS/narrow_peaks_dict.pkl'), 'wb'))


    # consensus peak: extend the summits half peak width, then overlap them and choose the dominant one
    from pycisTopic.iterative_peak_calling import get_consensus_peaks
    # Other param
    path_to_blacklist= '../third_party/data/hg38-blacklist.v2.bed'
    # Get consensus peaks
    consensus_peaks=get_consensus_peaks(narrow_peaks_dict, peak_half_width, chromsizes=chromsizes, path_to_blacklist=path_to_blacklist)

    # save
    consensus_peaks.to_bed(
        path = os.path.join(work_dir, 'scATAC/consensus_peak_calling/consensus_regions.bed'), 
        keep=True, 
        compression='infer', 
        chain=False)
import pandas as pd
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments.')
 # Define arguments
    parser.add_argument('--donor_id', type=str, help='Donor')
    # Parse arguments
    args = parser.parse_args()
    donor_id = args.donor_id

    work_dir = '../output/multiomics/'
    shift = 73
    ext_size = 146
    peak_half_width = 250
    fragments_dict = {donor_id: f'{work_dir}/fragments/{donor_id}.bed.gz'} # which version of fragments
    multiome_obs_meta = pd.read_csv('../data/multiome_obs_meta.csv')
    # cell_data = multiome_obs_meta[multiome_obs_meta.donor_id == 'donor_0'].set_index('obs_id').drop(columns=['donor_id']) # cell meta data 
    cell_data = multiome_obs_meta[multiome_obs_meta.donor_id == donor_id].set_index('obs_id')

    chromsizes = get_chrom_size()
    # pseudobulking
    pseudobulk(cell_data, chromsizes, fragments_dict)
    peak_calling()

  
    