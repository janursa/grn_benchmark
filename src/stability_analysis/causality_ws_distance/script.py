"""
Experiment: Causal Direction Test via Wasserstein Distance
==========================================================
Tests whether WS-distance favours causal (TF→gene) edges over
anti-causal (gene→TF) edges for selected GRN models.

Three chapters (each controlled by a flag):
  1. Compute all-gene background distribution
  2. Compute all-gene consensus (forward & reversed)
  3. Run the direction experiment
"""

import os
import sys
import random
import multiprocessing as mp

import numpy as np
import pandas as pd
import anndata as ad
import scipy.stats
from tqdm import tqdm

env = os.environ

sys.path.insert(0, env['UTILS_DIR'])
sys.path.insert(0, env['METRICS_DIR'])

from util import naming_convention, process_links, read_prediction

# ── runtime flags ──────────────────────────────────────────────────────────────
import argparse as _ap

_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument('--dataset', default='replogle')
_parser.add_argument('--test',    action='store_true')
_args, _ = _parser.parse_known_args()

DATASET  = _args.dataset
TEST_MODE = _args.test

ALL_MODELS = ['pearson_corr', 'spearman_corr', 'grnboost', 'ppcor', 'portia',
              'scenic', 'geneformer', 'scgpt', 'scprint',
              'positive_control', 'negative_control']
NUM_WORKERS = 40
SEED = 42

if TEST_MODE:
    ALL_MODELS  = ['pearson_corr', 'grnboost']
    NUM_WORKERS = 4
np.random.seed(SEED)
random.seed(SEED)

from src.params import get_par
par = get_par(DATASET)

rr_dir = f"{env['RESULTS_DIR']}/experiment/causality_direction/{DATASET}"
os.makedirs(rr_dir, exist_ok=True)
os.makedirs(f"{rr_dir}/tmp", exist_ok=True)

bg_path            = f"{rr_dir}/ws_distance_background_all.csv"
consensus_fwd_path = f"{rr_dir}/ws_consensus_all_forward.csv"
consensus_rev_path = f"{rr_dir}/ws_consensus_all_reversed.csv"
scores_path        = f"{rr_dir}/scores.csv"

COMPUTE_BACKGROUND = not os.path.exists(bg_path)
COMPUTE_CONSENSUS  = not (os.path.exists(consensus_fwd_path) and os.path.exists(consensus_rev_path))

# shared-memory path for the dense expression matrix (re-used across workers)
# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 1 – Background distribution for ALL perturbed genes
# ══════════════════════════════════════════════════════════════════════════════
_SC_PATH   = None   # set at runtime; workers read via module-level variable
_LAYER_MAP = ('X_norm', 'lognorm', 'pearson_residual')


def _load_adata(sc_path):
    """Load adata, set X from the first available layer, normalise pert labels."""
    adata = ad.read_h5ad(sc_path, backed='r').to_memory()
    for layer in _LAYER_MAP:
        if layer in adata.layers:
            adata.X = adata.layers[layer]
            break
    else:
        raise ValueError(f"None of {_LAYER_MAP} found in adata.layers")
    if 'is_control' not in adata.obs.columns:
        adata.obs['is_control'] = adata.obs['perturbation'] == 'non-targeting'
    adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)
    adata.obs.loc[adata.obs['is_control'], 'perturbation'] = 'control'
    return adata


# Module-level globals shared with workers via fork (no pickling of large arrays)
_worker_X_dense = None
_worker_gene_names = None
_worker_obs_perturbation = None


def _ws_for_gene_worker(perturbed_gene):
    """Worker: uses fork-inherited globals — no large-array pickling."""
    X_dense = _worker_X_dense
    gene_names = _worker_gene_names
    obs_perturbation = _worker_obs_perturbation

    ctrl_mask = obs_perturbation == 'control'
    pert_mask = obs_perturbation == perturbed_gene
    if pert_mask.sum() == 0:
        return pd.DataFrame(columns=['source', 'target', 'ws_distance'])

    # Pre-slice rows once per source — much more cache-friendly than per-column access
    X_ctrl = X_dense[ctrl_mask, :]   # (n_ctrl, n_genes)
    X_pert = X_dense[pert_mask, :]   # (n_pert, n_genes)

    rows = []
    for j, target in enumerate(gene_names):
        ws = scipy.stats.wasserstein_distance(X_ctrl[:, j], X_pert[:, j])
        rows.append({'source': perturbed_gene, 'target': target, 'ws_distance': ws})
    return pd.DataFrame(rows)


def compute_background(par, bg_path, num_workers=NUM_WORKERS):
    global _worker_X_dense, _worker_gene_names, _worker_obs_perturbation

    print("\n[Chapter 1] Computing all-gene background distribution …", flush=True)
    adata = _load_adata(par['evaluation_data_sc'])

    gene_names       = adata.var_names.to_numpy()
    obs_perturbation = adata.obs['perturbation'].to_numpy()

    # All perturbed genes that also appear in var_names (so we can score gene→target)
    available_genes = [g for g in np.unique(obs_perturbation)
                       if g != 'control' and g in set(gene_names)]

    # test-mode: ensure a mix of TF and non-TF sources so forward consensus is non-empty
    if TEST_MODE:
        tf_all_list = set(np.loadtxt(par['tf_all'], dtype=str))
        tf_sources  = [g for g in available_genes if g in tf_all_list][:15]
        nontf_sources = [g for g in available_genes if g not in tf_all_list][:15]
        available_genes = tf_sources + nontf_sources
        gene_names      = gene_names[:200]
        print(f"  [TEST MODE] {len(tf_sources)} TF + {len(nontf_sources)} non-TF sources, "
              f"{len(gene_names)} targets", flush=True)

    print(f"  Sources: {len(available_genes)}, Targets: {len(gene_names)}", flush=True)

    # Pre-extract the dense sub-matrix once (only target columns needed)
    gene_idx  = [np.where(adata.var_names == g)[0][0] for g in gene_names]
    X_dense   = adata.X[:, gene_idx]
    if hasattr(X_dense, 'toarray'):
        X_dense = X_dense.toarray()
    X_dense = np.array(X_dense, dtype=np.float32)

    # Set globals BEFORE creating Pool so fork-child processes inherit them
    _worker_X_dense = X_dense
    _worker_gene_names = gene_names
    _worker_obs_perturbation = obs_perturbation

    results = []
    if TEST_MODE or num_workers <= 1:
        for g in tqdm(available_genes, desc='Background (all genes)'):
            results.append(_ws_for_gene_worker(g))
    else:
        with mp.Pool(min(num_workers, len(available_genes))) as pool:
            for res in tqdm(pool.imap(_ws_for_gene_worker, available_genes),
                            total=len(available_genes), desc='Background (all genes)'):
                results.append(res)

    bg = pd.concat(results, ignore_index=True)
    bg.to_csv(bg_path, index=False)
    print(f"  Saved → {bg_path}  ({len(bg):,} rows)", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 2 – Consensus for all-gene forward & reversed GRNs
# ══════════════════════════════════════════════════════════════════════════════
def _load_grns(par, models, models_dir):
    grn_store = []
    for model in models:
        path = f"{models_dir}/{naming_convention(DATASET, model)}"
        if not os.path.exists(path):
            print(f"  [skip] {path} not found", flush=True)
            continue
        grn = ad.read_h5ad(path).uns['prediction']
        grn = process_links(grn, par)
        grn['model'] = model
        grn_store.append(grn)
    return pd.concat(grn_store, ignore_index=True)


def _build_consensus(grn_all, consensus_path, available_sources=None):
    """Compute quantile consensus (theta=0.25/0.75) over n_edges per source."""
    if available_sources is not None:
        grn_all = grn_all[grn_all['source'].isin(available_sources)]

    edges_count = (grn_all
                   .groupby(['source', 'model'])
                   .size()
                   .reset_index(name='n_edges')
                   .pivot(index='source', columns='model')
                   .fillna(0))

    rows = []
    for src, row in edges_count.iterrows():
        row_nz = row[row != 0]
        if len(row_nz) == 0:
            continue
        rows.append({'source': src, 'theta': 0.25, 'value': int(np.quantile(row_nz, 0.25))})
        rows.append({'source': src, 'theta': 0.75, 'value': int(np.quantile(row_nz, 0.75))})

    consensus = pd.DataFrame(rows)
    consensus.to_csv(consensus_path, index=False)
    print(f"  Saved consensus ({len(consensus)//2} sources) → {consensus_path}", flush=True)
    return consensus


def compute_consensus(par, models, models_dir,
                      consensus_fwd_path, consensus_rev_path,
                      bg_path):
    print("\n[Chapter 2] Computing all-gene consensus …", flush=True)
    bg_sources = pd.read_csv(bg_path, usecols=['source'])['source'].unique()

    grn_all = _load_grns(par, models, models_dir)

    # Forward consensus – any gene that is both a source in GRNs and in background
    print("  Building forward consensus …", flush=True)
    _build_consensus(grn_all, consensus_fwd_path, available_sources=bg_sources)

    # Reversed consensus – swap source↔target, then build consensus
    print("  Building reversed consensus …", flush=True)
    grn_rev = grn_all.rename(columns={'source': 'target', 'target': 'source'})
    _build_consensus(grn_rev, consensus_rev_path, available_sources=bg_sources)


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 3 – Direction experiment
# ══════════════════════════════════════════════════════════════════════════════
def _score_grn(net, consensus, bg_by_source, label, seed=SEED):
    """
    Score a GRN against the all-gene background.
    Returns a DataFrame with per-TF/gene ws_distance_pc values and summary.

    bg_by_source: dict {source -> DataFrame} pre-indexed from background_distance.
    """
    np.random.seed(seed)
    bg_sources = np.array(list(bg_by_source.keys()))
    consensus_sources = consensus['source'].unique()

    prediction_sources = set(net['source'].unique())
    sources_to_eval = np.intersect1d(np.intersect1d(list(prediction_sources), bg_sources), consensus_sources)

    # Pre-index consensus by theta and source for O(1) lookup
    consensus_idx = {
        theta: consensus_theta.set_index('source')['value']
        for theta, consensus_theta in consensus.groupby('theta')
    }
    # Pre-index net by source for O(1) lookup
    net_by_source = {src: grp for src, grp in net.groupby('source')}

    scores_store = []
    for src in sources_to_eval:
        bg_src = bg_by_source[src]
        bg_ws_vals = bg_src['ws_distance'].values
        # Sample bg_rand once per source (not per theta)
        bg_rand_sorted = np.sort(np.random.choice(bg_ws_vals, 1000))

        for theta, c_idx in consensus_idx.items():
            if src not in c_idx.index:
                continue
            n_edges = int(c_idx[src])

            if src in prediction_sources:
                net_src = net_by_source[src].nlargest(n_edges, 'weight')
                ws = bg_src[bg_src['target'].isin(net_src['target'])].copy()
            else:
                ws = bg_src.sample(n_edges, replace=True, random_state=seed).copy()

            # fill missing links with random background draws
            n_missing = n_edges - len(ws)
            if n_missing > 0:
                ws = pd.concat([ws, bg_src.sample(n_missing, replace=True, random_state=seed)])

            # Vectorized percentile ranking via searchsorted
            ws = ws.copy()
            raw_ws = ws['ws_distance'].values
            ws['ws_distance_pc'] = np.searchsorted(bg_rand_sorted, raw_ws) / len(bg_rand_sorted)
            # Raw WS stored per row for downstream raw-score analysis
            ws['bg_ws_mean'] = bg_ws_vals.mean()
            ws['source'] = src
            ws['theta'] = theta
            ws['direction'] = label
            scores_store.append(ws)

    if not scores_store:
        return pd.DataFrame(), pd.DataFrame()

    detailed = pd.concat(scores_store, ignore_index=True)

    def _agg_summary(g):
        # Two-level aggregation: per-source mean first, then median across sources.
        # This gives each source equal weight regardless of edge count, and the
        # median makes the final score robust to pleiotropic outlier sources.
        per_src_pc  = g.groupby(['source', 'theta'])['ws_distance_pc'].mean().reset_index()
        per_src_raw = g.groupby(['source', 'theta'])['ws_distance'].mean().reset_index()
        per_src_bg  = g.groupby('source')['bg_ws_mean'].first()

        p_pc  = per_src_pc[per_src_pc['theta'] == 0.25]['ws_distance_pc'].median()
        r_pc  = per_src_pc[per_src_pc['theta'] == 0.75]['ws_distance_pc'].median()
        p_raw = per_src_raw[per_src_raw['theta'] == 0.25]['ws_distance'].median()
        r_raw = per_src_raw[per_src_raw['theta'] == 0.75]['ws_distance'].median()
        bg_m  = per_src_bg.median()
        f1_pc  = 2*p_pc*r_pc/(p_pc+r_pc)   if (p_pc+r_pc)   > 0 else 0.0
        f1_raw = 2*p_raw*r_raw/(p_raw+r_raw) if (p_raw+r_raw) > 0 else 0.0
        return pd.Series({
            'ws_precision':     p_pc,
            'ws_recall':        r_pc,
            'ws_f1':            f1_pc,
            'ws_precision_raw': p_raw,
            'ws_recall_raw':    r_raw,
            'ws_f1_raw':        f1_raw,
            'bg_ws_median':     bg_m,
            'ws_ratio':         (p_raw + r_raw) / 2 / bg_m if bg_m > 0 else np.nan,
            'n_sources':        g['source'].nunique(),
        })

    summary = _agg_summary(detailed).to_frame().T.reset_index(drop=True)
    return detailed, summary


def run_experiment(par, models, models_dir,
                   bg_path, consensus_fwd_path, consensus_rev_path,
                   scores_path):
    print("\n[Chapter 3] Running direction experiment …", flush=True)

    background = pd.read_csv(bg_path)
    # Pre-index background by source once — avoids O(29M) scan per lookup
    bg_by_source = {src: grp.reset_index(drop=True)
                    for src, grp in background.groupby('source')}

    def _safe_read_consensus(path):
        try:
            df = pd.read_csv(path)
            if df.empty or 'source' not in df.columns:
                raise ValueError("empty")
            return df
        except Exception:
            print(f"  Warning: consensus file empty or missing: {path}", flush=True)
            return pd.DataFrame(columns=['source', 'theta', 'value'])

    consensus_fwd = _safe_read_consensus(consensus_fwd_path)
    consensus_rev = _safe_read_consensus(consensus_rev_path)

    # Genes to exclude from reversed sources:
    #   (a) Core essential genes (Hart et al. CEGv2) — their perturbation causes
    #       global transcriptomic shifts unrelated to specific TF regulation.
    #   (b) TF genes — the reversed experiment should only test non-TF → TF edges.
    #   High-pleiotropy sources are handled by the median aggregation in _score_grn
    #   (per-source mean → median across sources), so no hard threshold filter needed.
    tfs_set = set(np.loadtxt(par['tf_all'], dtype=str))
    ceg_set = set(pd.read_csv(par['ceg'], sep='\t', usecols=['GENE'])['GENE'].dropna())

    rev_exclude = tfs_set | ceg_set
    n_ceg = len(ceg_set); n_tf = len(tfs_set)
    print(f"  Reversed exclusions: {n_ceg} CEGv2 essential + {n_tf} TFs "
          f"= {len(rev_exclude)} unique genes excluded", flush=True)

    all_scores = []
    for model in models:
        path = f"{models_dir}/{naming_convention(DATASET, model)}"
        if not os.path.exists(path):
            print(f"  [skip] {model} – prediction file not found", flush=True)
            continue
        print(f"  Evaluating {model} …", flush=True)

        grn_obj = ad.read_h5ad(path)
        net = grn_obj.uns['prediction']
        par_tmp = {**par, 'max_n_links': 50_000, 'verbose': 0}
        net = process_links(net, par_tmp)

        # Forward (TF→gene)
        _, summary_fwd = _score_grn(net, consensus_fwd, bg_by_source, label='forward')
        if len(summary_fwd) > 0:
            summary_fwd['model']     = model
            summary_fwd['direction'] = 'forward'
            summary_fwd['dataset']   = DATASET
            all_scores.append(summary_fwd)

        # Reversed (non-TF, non-essential gene → TF):
        # Swap source↔target, then drop sources that are TFs or core essential genes.
        net_rev = net.rename(columns={'source': 'target', 'target': 'source'})
        net_rev['weight'] = net['weight'].values  # preserve original weight
        n_before = net_rev['source'].nunique()
        net_rev = net_rev[~net_rev['source'].isin(rev_exclude)]
        n_after = net_rev['source'].nunique()
        print(f"    {model} reversed: {n_before} → {n_after} sources after exclusions", flush=True)

        _, summary_rev = _score_grn(net_rev, consensus_rev, bg_by_source, label='reversed')
        if len(summary_rev) > 0:
            summary_rev['model']     = model
            summary_rev['direction'] = 'reversed'
            summary_rev['dataset']   = DATASET
            all_scores.append(summary_rev)

    if all_scores:
        scores_df = pd.concat(all_scores, ignore_index=True)
        scores_df.to_csv(scores_path, index=False)
        print(f"  Saved scores → {scores_path}", flush=True)
    else:
        print("  Warning: no scores produced.", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    models_dir = f"{env['RESULTS_DIR']}/{DATASET}"
    print(f"\n[Dataset: {DATASET}]  bg_exists={not COMPUTE_BACKGROUND}  "
          f"consensus_exists={not COMPUTE_CONSENSUS}", flush=True)

    if COMPUTE_BACKGROUND:
        compute_background(par, bg_path, num_workers=NUM_WORKERS)

    if COMPUTE_CONSENSUS:
        compute_consensus(par, ALL_MODELS, models_dir,
                          consensus_fwd_path, consensus_rev_path,
                          bg_path)

    run_experiment(par, ALL_MODELS, models_dir,
                   bg_path, consensus_fwd_path, consensus_rev_path,
                   scores_path)
