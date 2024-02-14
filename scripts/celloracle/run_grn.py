import pandas as pd
import os 
from celloracle import motif_analysis as ma
import pandas as pd
import celloracle as co

oracle = co.load_hdf5('../output/celloracle/grn/presaved.celloracle.oracle')
# This step may take some time.
links = oracle.get_links(cluster_name_for_GRN_unit="cell_type", alpha=10,
                        verbose_level=10)
links.to_hdf5(file_path="../output/celloracle/grn/links.celloracle.links")