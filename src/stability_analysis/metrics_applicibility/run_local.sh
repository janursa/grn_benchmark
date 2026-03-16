#!/bin/bash
# Run metric applicability evaluation in local mode.
# Usage: bash src/stability_analysis/metrics_applicibility/run_local.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source env.sh
python scripts/evaluate_metric_applicability.py --local_run "$@"
