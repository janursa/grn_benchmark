# Metric Quality Evaluation Script

## Overview
This script evaluates which metrics should be kept for each dataset based on two criteria:
1. **Variability (CV)**: The coefficient of variation (std/mean) must meet a threshold (default: 0.2)
2. **Meaningful Max Score**: The maximum score obtained must exceed a metric-specific threshold

## Script Location
`/home/jnourisa/projs/ongoing/grn_benchmark/scripts/evaluate_metric_quality.py`

## Usage

### Basic usage (evaluate all datasets):
```bash
cd /home/jnourisa/projs/ongoing/grn_benchmark
source ~/.bash_profile && conda activate py10 && source env.sh
python scripts/evaluate_metric_quality.py
```

### Evaluate specific datasets:
```bash
python scripts/evaluate_metric_quality.py --datasets op parsebioscience replogle
```

### Adjust CV threshold:
```bash
python scripts/evaluate_metric_quality.py --cv_threshold 0.15
```

### Specify output file:
```bash
python scripts/evaluate_metric_quality.py --output /path/to/output.csv
```

### Evaluate specific metrics:
```bash
python scripts/evaluate_metric_quality.py --metrics r_precision r_recall vc
```

## Metric-Specific Thresholds

Based on analysis of metric implementation in `task_grn_inference/src/metrics/`, the following thresholds were defined:

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **r_precision** | 0.1 | R²-based regression metric; >0.1 indicates meaningful predictive power |
| **r_recall** | 0.1 | R²-based regression metric; >0.1 indicates meaningful predictive power |
| **ws_precision** | 0.05 | Wasserstein distance-based; lower threshold due to metric nature |
| **ws_recall** | 0.05 | Wasserstein distance-based; lower threshold due to metric nature |
| **vc** | 0.3 | Virtual cell correlation; >0.3 indicates moderate correlation |
| **sem** | 0.4 | Structural Equation Modeling fit; >0.4 indicates acceptable fit |
| **t_rec_precision** | 0.1 | TF recovery precision; >0.1 indicates meaningful TF identification |
| **t_rec_recall** | 0.1 | TF recovery recall; >0.1 indicates meaningful TF identification |
| **replicate_consistency** | 0.3 | Replica consistency correlation; >0.3 indicates good consistency |
| **tfb_f1** | 0.05 | TF binding F1; lower threshold due to difficulty of task |
| **gs_f1** | 0.1 | Gene set recovery F1; >0.1 indicates meaningful enrichment |

## Results Summary

### Overall Statistics
- **Total combinations**: 110 (11 metrics × 10 datasets)
- **Metrics computed**: 66 (60.0%)
- **Metrics to keep**: 37 (33.6%)

### Key Findings

**Metrics that consistently pass** (across most datasets):
- `r_precision`: 9/10 datasets ✓
- `gs_f1`: 9/10 datasets ✓
- `tfb_f1`: 8/9 datasets ✓

**Metrics that never pass**:
- `sem`: 0/7 datasets (max scores too low, typically <0.2 vs threshold of 0.4)
- `vc`: 0/3 datasets (low variability CV<0.2)
- `ws_precision`: 0/4 datasets (low variability)
- `ws_recall`: 0/4 datasets (low variability)

**Metrics with moderate success**:
- `replicate_consistency`: 2/3 datasets
- `t_rec_precision`: 2/3 datasets
- `t_rec_recall`: 2/3 datasets

### Recommended Metrics per Dataset

**op**: `r_precision`, `replicate_consistency`, `tfb_f1`, `gs_f1`
**parsebioscience**: `r_precision`, `replicate_consistency`, `tfb_f1`, `gs_f1`
**300BCG**: `r_precision`, `replicate_consistency`, `tfb_f1`, `gs_f1`
**ibd_uc**: `r_precision`, `tfb_f1`, `gs_f1`
**ibd_cd**: `r_precision`, `tfb_f1`, `gs_f1`
**replogle**: `r_precision`, `r_recall`, `t_rec_precision`, `t_rec_recall`, `tfb_f1`, `gs_f1`
**xaira_HEK293T**: `r_precision`, `t_rec_precision`, `t_rec_recall`, `gs_f1`
**xaira_HCT116**: `r_precision`, `t_rec_precision`, `t_rec_recall`, `tfb_f1`, `gs_f1`
**nakatake**: `r_precision`, `r_recall`, `gs_f1`
**norman**: `tfb_f1` (only 1 metric passes!)

## Output File

Results are saved to:
`/home/jnourisa/projs/ongoing/task_grn_inference/resources/results/metric_quality_evaluation.csv`

### CSV Columns:
- `dataset`: Dataset name
- `metric`: Metric name
- `keep`: Boolean - whether to keep this metric for this dataset
- `present`: Boolean - whether metric was computed
- `reason`: Explanation for the decision
- `n_methods`: Number of methods with scores
- `mean`: Mean score across methods
- `std`: Standard deviation
- `cv`: Coefficient of variation (std/mean)
- `max`: Maximum score
- `threshold`: Metric-specific threshold used
- `cv_threshold`: CV threshold used

## Interpretation Guidelines

### Why drop a metric?

1. **Low variability**: CV < 0.2 means all methods perform similarly, so the metric doesn't discriminate between good and bad methods
2. **Low max score**: Even the best method doesn't reach a meaningful performance level

### Why keep a metric?

Both conditions must be met:
1. CV ≥ 0.2 (methods show sufficient variation)
2. max ≥ threshold (at least one method achieves meaningful performance)

## Additional Considerations

### SEM (Structural Equation Modeling)
- Consistently fails (max <0.2 vs threshold 0.4)
- Consider either:
  - Lowering threshold to 0.15-0.2
  - Investigating why scores are universally low
  - Removing from analysis

### VC (Virtual Cell)
- Fails due to low variability (all methods ~0.3-0.4)
- Either all methods are similarly good/bad at this task
- Consider if this metric is too easy/hard for current methods

### Norman dataset
- Only 1 metric passes (tfb_f1)
- Suggests this dataset may need additional metrics or the existing ones aren't suitable
