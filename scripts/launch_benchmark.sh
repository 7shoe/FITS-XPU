#!/usr/bin/env bash
# Run the standard five-seed, four-horizon benchmark as independent experiments
# on a bounded pool of Aurora XPU tiles.

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

if [[ "$model" != all && ! -f "scripts/${model}/aurora/run_selected.sh" ]]; then
    echo "No implemented Aurora launcher for model ${model}." >&2
    exit 2
fi

# Override either list for a shorter controlled run, e.g.
# MODEL_SEEDS='514' MODEL_HORIZONS='96'.
seeds=( ${MODEL_SEEDS:-114 514 1919 810 0} )
horizons=( ${MODEL_HORIZONS:-96 192 336 720} )

if [[ "${MODEL_DRY_RUN:-0}" != 1 ]]; then
    if [[ -n "${ZE_AFFINITY_MASK:-}" ]]; then
        echo "Experiment pooling requires the full node, but ZE_AFFINITY_MASK is already set to '${ZE_AFFINITY_MASK}'." >&2
        echo "Unset it before launching; the pool assigns one flat tile to each child." >&2
        exit 2
    fi

    source /home/siebenschuh/Projects/Aurora_HPC/env_tsfm/activate_tsfm_venv.sh
    export MODEL_ENV_READY=1
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT
    export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

    if [[ -z "${MODEL_XPU_TILES:-}" ]]; then
        # xpu-smi reports Aurora's six physical cards. FLAT exposes two tiles
        # from each card, giving 12 independent PyTorch devices, without
        # creating a disposable PyTorch/Level Zero training context here.
        if ! visible_xpus="$(xpu-smi discovery -j | python -c '
import json, sys
payload = json.load(sys.stdin)
print(2 * len(payload.get("device_list") or []))
' 2>/dev/null)"; then
            echo "Could not query Aurora devices with xpu-smi." >&2
            exit 1
        fi
        if [[ "$visible_xpus" -lt 1 ]]; then
            echo "No XPU tiles detected; run this command inside an Aurora compute-node allocation." >&2
            exit 1
        fi
        MODEL_XPU_TILES="$(seq -s, 0 $((visible_xpus - 1)))"
        export MODEL_XPU_TILES
    fi
fi

# Unless explicitly capped, admit one independent experiment for every flat
# XPU device assigned to the pool.  Keep range parsing here consistent with
# run_benchmark_pool.py so overrides such as MODEL_XPU_TILES='0-5' work too.
if [[ -z "${MODEL_MAX_PARALLEL:-}" ]]; then
    tile_spec="${MODEL_XPU_TILES:-0-11}"
    MODEL_MAX_PARALLEL="$(python - "$tile_spec" <<'PY'
import re
import sys

tiles = []
for token in re.split(r"[\s,]+", sys.argv[1].strip()):
    if not token:
        continue
    match = re.fullmatch(r"(\d+)-(\d+)", token)
    if match:
        first, last = map(int, match.groups())
        if last < first:
            raise SystemExit("descending XPU range: {}".format(token))
        tiles.extend(range(first, last + 1))
    elif token.isdigit():
        tiles.append(int(token))
    else:
        raise SystemExit("invalid flat XPU tile: {}".format(token))
if not tiles or len(set(tiles)) != len(tiles):
    raise SystemExit("XPU tile list must be non-empty and distinct")
print(len(tiles))
PY
)"
    export MODEL_MAX_PARALLEL
fi

# Pooled runs avoid multiplying DataLoader subprocesses across 12 experiments.
export MODEL_NUM_WORKERS="${MODEL_NUM_WORKERS:-0}"
export MODEL_CPU_THREADS="${MODEL_CPU_THREADS:-4}"
export MODEL_SKIP_COMPLETED="${MODEL_SKIP_COMPLETED:-1}"
horizon_csv="$(IFS=,; echo "${horizons[*]}")"
seed_csv="$(IFS=,; echo "${seeds[*]}")"

exec python -u scripts/run_benchmark_pool.py \
    --horizons "$horizon_csv" \
    --seeds "$seed_csv" \
    "$model" "$action" "$@"
