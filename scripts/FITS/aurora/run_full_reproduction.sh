#!/usr/bin/env bash
# Run an isolated FITS five-seed reproduction from training through evaluation
# and table generation. Re-running the same tag resumes successful settings.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/FITS/aurora/run_full_reproduction.sh [all|train|test] [DATASET ...]

Defaults to all phases and all seven long-term forecasting datasets.
Override FITS_REPRO_TAG to create a separate reproduction lane.
EOF
}

phase="${1:-all}"
case "$phase" in
    all|train|test)
        if [[ $# -gt 0 ]]; then shift; fi
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown phase: $phase" >&2
        usage >&2
        exit 2
        ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

if [[ $# -gt 0 ]]; then
    datasets=("$@")
else
    datasets=(ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic)
fi

storage_root="/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining"
repro_tag="${FITS_REPRO_TAG:-parallel_reproduction_01}"
repro_root="${FITS_REPRO_ROOT:-$storage_root/FITS_reproductions/$repro_tag}"

export MODEL_HORIZONS="${MODEL_HORIZONS:-96 192 336 720}"
export MODEL_SEEDS="${MODEL_SEEDS:-114 514 1919 810 0}"
export MODEL_SKIP_COMPLETED="${MODEL_SKIP_COMPLETED:-1}"
export MODEL_NUM_WORKERS="${MODEL_NUM_WORKERS:-0}"
export MODEL_CPU_THREADS="${MODEL_CPU_THREADS:-4}"
export FITS_CHECKPOINTS_ROOT="${FITS_CHECKPOINTS_ROOT:-$repro_root/FITS_checkpoints/FITS}"
export FITS_RESULTS_ROOT="${FITS_RESULTS_ROOT:-$repro_root/FITS_results/FITS}"
export MODEL_LOGS_ROOT="${MODEL_LOGS_ROOT:-$repro_root/logs}"

mkdir -p "$FITS_CHECKPOINTS_ROOT" "$FITS_RESULTS_ROOT" "$MODEL_LOGS_ROOT"

active_pid=""
received_signal=""

forward_to_pool() {
    local signal_name="$1"
    received_signal="$signal_name"
    echo "Reproduction wrapper received ${signal_name}; forwarding it to the experiment pool." >&2
    if [[ -n "$active_pid" ]] && kill -0 "$active_pid" 2>/dev/null; then
        kill -s "$signal_name" "$active_pid" 2>/dev/null || true
    fi
}

trap 'forward_to_pool TERM' TERM
trap 'forward_to_pool INT' INT

run_phase() {
    local requested_phase="$1"
    local rc

    set +e
    bash scripts/launch_benchmark.sh FITS "$requested_phase" "${datasets[@]}" &
    active_pid=$!
    while true; do
        wait "$active_pid"
        rc=$?
        if ! kill -0 "$active_pid" 2>/dev/null; then
            break
        fi
    done
    active_pid=""
    set -e

    if [[ -n "$received_signal" ]]; then
        return 1
    fi
    return "$rc"
}

cat <<EOF
FITS reproduction: $repro_tag
Phase:             $phase
Datasets:          ${datasets[*]}
Horizons:          $MODEL_HORIZONS
Seeds:             $MODEL_SEEDS
Parallelism:       ${MODEL_MAX_PARALLEL:-one experiment per detected XPU}
Checkpoints:       $FITS_CHECKPOINTS_ROOT
Results:           $FITS_RESULTS_ROOT
Logs:              $MODEL_LOGS_ROOT
EOF

if [[ "$phase" == all || "$phase" == train ]]; then
    run_phase train
fi

if [[ "$phase" == all || "$phase" == test ]]; then
    run_phase test

    metrics_csv="$repro_root/model_metrics.csv"
    summary_csv="$repro_root/model_summary.csv"
    summary_md="$repro_root/model_summary.md"
    python scripts/merge_model_metrics.py \
        --search-root "$repro_root" \
        --output "$metrics_csv"
    python scripts/summarize_model_metrics.py \
        --input "$metrics_csv" \
        --output "$summary_csv" \
        --markdown-output "$summary_md"

    echo "FITS comparison table: $summary_md"
fi

echo "FITS reproduction complete: $repro_root"
