#!/usr/bin/env bash
# Run the standard five-seed, four-horizon benchmark sequentially for one model
# lane or every implemented lane.

set -e -o pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 {FITS|DIFFS|ANG|ROUTER|all} {train|test} DATASET [DATASET ...]" >&2
    exit 2
fi

model="$1"
action="$2"
shift 2

case "$action" in
    train|test) ;;
    *) echo "action must be train or test" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Override either list for a shorter controlled run, e.g.
# MODEL_SEEDS='514' MODEL_HORIZONS='96'.
seeds=( ${MODEL_SEEDS:-114 514 1919 810 0} )
horizons=( ${MODEL_HORIZONS:-96 192 336 720} )

for dataset in "$@"; do
    for horizon in "${horizons[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "===== ${model} ${action}: ${dataset}, H=${horizon}, seed=${seed} ====="
            bash scripts/launch_model.sh "$model" "$action" "$dataset" "$horizon" "$seed"
        done
    done
done
