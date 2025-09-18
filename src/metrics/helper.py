
import os
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import sys
import os
env = os.environ
TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
RESULTS_DIR = env['RESULTS_DIR']

sys.path.append(env["METRICS_DIR"])
sys.path.append(env["UTILS_DIR"])

from regression_2.helper import main as main_reg2
from ws_distance.helper import main as main_ws_distance
from sem.helper import main as main_sem


def metrics_all(par):
    rr_reg2 = main_reg2(par)
    _, rr_ws = main_ws_distance(par)
    rr_sem = main_sem(par)

    rr_all = pd.concat([rr_reg2, rr_ws, rr_sem], axis=1)
    return rr_all