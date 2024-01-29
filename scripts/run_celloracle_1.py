import pandas as pd
import os 
from celloracle import motif_analysis as ma
import pandas as pd

donor_id = 'donor_0'
cell_type = 'B cells'


peaks = pd.read_csv(f"all_peaks_donor_0_B cells.csv", index_col=0)
cicero_connections =  pd.read_csv(f"connections_{donor_id}_{cell_type}.csv", index_col=0)


tss_annotated = ma.get_tss_info(peak_str_list=peaks['x'].values, ref_genome="hg38")

integrated = ma.integrate_tss_peak_with_cicero(tss_peak=tss_annotated, 
                                               cicero_connections=cicero_connections)

processed_peak = integrated[integrated.coaccess>0.8].reset_index(drop=True)

# PLEASE make sure reference genome is correct.
ref_genome = "hg38"

genome_installation = ma.is_genome_installed(ref_genome=ref_genome,
                                             genomes_dir=None)
print(ref_genome, "installation: ", genome_installation)


# Instantiate TFinfo object
tfi = ma.TFinfo(peak_data_frame=processed_peak, 
                ref_genome="hg38",
                genomes_dir=None) 

tfi.scan(fpr=0.05, 
         motifs=None,  # If you enter None, default motifs will be loaded.
         verbose=True)

# Reset filtering 
tfi.reset_filtering()

# Do filtering
tfi.filter_motifs_by_score(threshold=10)

# Format post-filtering results.
tfi.make_TFinfo_dataframe_and_dictionary(verbose=True)


df = tfi.to_dataframe()
df.to_csv('grn_celloracle_base.csv')