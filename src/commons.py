import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import matplotlib
# work_dir = '../output'

matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex='false')

# def format_folder(work_dir, exclude_missing_genes, reg_type, theta, tf_n, norm_method, subsample=None):
#     return f'{work_dir}/benchmark/scores/subsample_{subsample}/exclude_missing_genes_{exclude_missing_genes}/{reg_type}/theta_{theta}_tf_n_{tf_n}/{norm_method}'

batch_key = 'plate_name'
label_key = 'cell_type'

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
    'GRaNIE': '#009E73',
    'ANANSE': '#e2b1cd',
    'scGLUE': '#D55E00',
    'Scenic+': '#dfc2e5',
    'HKGs': 'darkblue',
    'Positive': 'darkblue',
    'Negative': 'indianred',
    'Positive Control': 'darkblue',
    'Negative Control': 'indianred',
    'GRNBoost2': '#e2b1cd',
    'GENIE3': '#009E73',
    'Pearson corr.': '#56B4E9',
    
}

LINESTYLES = {
    'Positive Control': '-',
    'Negative Control': '-',
    'CollectRI': '--',
    'FigR': '-.',
    'CellOracle': ':',
    'GRaNIE': ':',
    'ANANSE': '--',
    'scGLUE': '-.',
    'Scenic+': '-',
    'GRNBoost2': '--',
    'GENIE3': '-.',
    'Pearson corr.': '-' 
}

MARKERS = {
    'Random': '.',
    'CollectRI': '.',
    'FigR': 'o',
    'CellOracle': 'd',
    'GRaNIE': 'v',
    'ANANSE': 's',
    'scGLUE': 'x',
    'Scenic+': '*',
    'Positive Control': '.',
    'Negative Control': '.',
    'GRNBoost2': 'x',
    'GENIE3': 'v',
    'Pearson corr.': '-' 
}

surragate_names = {'CollectRI': 'CollectRI', 'collectRI':'CollectRI', 'collectRI_sign':'CollectRI-signs', 'collectri': 'CollectRI',
                   'Scenic+': 'Scenic+', 'scenicplus':'Scenic+', 'scenicplus_sign': 'Scenic+-signs',
                   'CellOracle': 'CellOracle', 'celloracle':'CellOracle', 'celloracle_sign':'CellOracle-signs',
                   'figr':'FigR', 'figr_sign':'FigR-signs',
                   'genie3': 'GENIE3',
                   'grnboost2':'GRNboost2',
                   'ppcor':'PPCOR',
                   'portia':'Portia',
                   'baseline':'Baseline',
                   'cov_net': 'Pearson cov',
                   'granie':'GRaNIE',
                   'ananse':'ANANSE',
                   'scglue':'scGLUE',
                   'pearson_corr': 'Pearson corr.',
                   'grnboost2': 'GRNBoost2',
                   'genie3': 'GENIE3',
                   'scenic': 'Scenic',
                   
                   'positive_control':'Positive Control',
                   'negative_control':'Negative Control',
                   'pearson':'APR',
                   'SL':'SLA',
                   'lognorm':'SLA',
                   'seurat_pearson': 'Seurat-APR',
                   'seurat_lognorm': 'Seurat-SLA',
                   'scgen_lognorm': 'scGEN-SLA',
                   'scgen_pearson': 'scGEN-APR',
                   'static-theta-0.0': r'$\theta$=min', 
                    'static-theta-0.5': r'$\theta$=median', 
                    'static-theta-1.0': r'$\theta$=max',
                    'S1': 'S1',
                    'S2': 'S2'
                   }
controls3 = ['Dabrafenib', 'Belinostat', 'Dimethyl Sulfoxide']
CELL_TYPES = ['NK cells', 'T cells CD4+', 'T cells CD8+', 'T regulatory cells', 'B cells', 'Myeloid cells']
negative_control = 'Dimethyl Sulfoxide'
controls2 = ['Dabrafenib', 'Belinostat']
T_cell_types = ['T regulatory cells', 'T cells CD8+', 'T cells CD4+']
cell_type_map = {cell_type: 'T cells' if cell_type in T_cell_types else cell_type for cell_type in CELL_TYPES}
cell_types = ['NK cells', 'T cells', 'B cells', 'Myeloid cells']


if False:
    collectRI = pd.read_csv("https://github.com/pablormier/omnipath-static/raw/main/op/collectri-26.09.2023.zip")
    collectRI.to_csv(f'{work_dir}/collectri.csv')




colors_cell_type = ['#c4d9b3', '#c5bc8e', '#c49e81', '#c17d88', 'gray', 'lightsteelblue']


colors_positive_controls = ['blue', 'cyan']
