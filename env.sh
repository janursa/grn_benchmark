# env.sh
export TASK_GRN_INFERENCE_DIR="/home/jnourisa/projs/ongoing/task_grn_inference"
export GRN_BENCHMARK_DIR="/home/jnourisa/projs/ongoing/grn_benchmark"
export PYTHONPATH="$GRN_BENCHMARK_DIR:${PYTHONPATH:-}"

export RESULTS_DIR="${TASK_GRN_INFERENCE_DIR}/resources/results"
export IMAGES_DIR="/home/jnourisa/projs/images"
export RESOURCES_DIR="${TASK_GRN_INFERENCE_DIR}/resources"
export INFERENCE_DIR="${RESOURCES_DIR}/grn_benchmark/inference_data"
export EVALUATION_DIR="${RESOURCES_DIR}/grn_benchmark/evaluation_data"
export PRIOR_DIR="${RESOURCES_DIR}/grn_benchmark/prior"
export EXTENDED_DIR="${RESOURCES_DIR}/extended_data"
export METHODS_DIR="${TASK_GRN_INFERENCE_DIR}/src/methods"
export METRICS_DIR="${TASK_GRN_INFERENCE_DIR}/src/metrics"
export UTILS_DIR="${TASK_GRN_INFERENCE_DIR}/src/utils"

# echo "Environment variables set:"
# echo "TASK_GRN_INFERENCE_DIR=$TASK_GRN_INFERENCE_DIR"
# echo "GRN_BENCHMARK_DIR=$GRN_BENCHMARK_DIR"
# echo "RESULTS_DIR=$RESULTS_DIR"
# echo "IMAGES_DIR=$IMAGES_DIR"
# echo "RESOURCES_DIR=$RESOURCES_DIR"
# echo "INFERENCE_DIR=$INFERENCE_DIR"
# echo "EVALUATION_DIR=$EVALUATION_DIR"
# echo "PRIOR_DIR=$PRIOR_DIR"
# echo "EXTENDED_DIR=$EXTENDED_DIR"
# echo "METHODS_DIR=$METHODS_DIR"
# echo "METRICS_DIR=$METRICS_DIR"
# echo "UTILS_DIR=$UTILS_DIR"


# -------- Generate config.yaml from this env.sh --------
output="env.yaml"
echo "# Auto-generated from env.sh" > "$output"

# Iterate over exported variables in this script
for var in TASK_GRN_INFERENCE_DIR GRN_BENCHMARK_DIR PYTHONPATH RESULTS_DIR IMAGES_DIR RESOURCES_DIR INFERENCE_DIR EVALUATION_DIR PRIOR_DIR EXTENDED_DIR METHODS_DIR METRICS_DIR UTILS_DIR; do
    value="$(eval echo "\"\$$var\"")"
    echo "$var: \"$value\"" >> "$output"
done

echo "Wrote $output"