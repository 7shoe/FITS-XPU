#!/usr/bin/env bash
# Model-explicit, experiment-parallel corpus pretraining on one Aurora node.
set -e -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -eq 0 || "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/launch_pretrain.sh MODEL [MODEL ...] train {v1|v2|v3} [...]'
    echo 'Each model/corpus/horizon/seed is a separate checkpointed experiment.'
    exit 0
fi
# Check the requested model registry before expensive environment/device setup.
python3 -m foundation.pretrain.launch --check "$@"
if [[ "${MODEL_DRY_RUN:-0}" != 1 ]]; then
    if [[ -n "${ZE_AFFINITY_MASK:-}" ]]; then
        echo 'Unset ZE_AFFINITY_MASK; the pretraining pool assigns tiles.' >&2
        exit 2
    fi
    source /home/siebenschuh/Projects/Aurora_HPC/env_tsfm/activate_tsfm_venv.sh
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT
    export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
    if [[ -z "${MODEL_XPU_TILES:-}" ]]; then
        MODEL_XPU_TILES="$(xpu-smi discovery -j | python -c '
import json, sys
n = 2 * len(json.load(sys.stdin).get("device_list") or [])
if not n: raise SystemExit("No Aurora GPUs detected; use a compute allocation")
print(",".join(map(str, range(n))))
')"
        export MODEL_XPU_TILES
    fi
fi
export MODEL_CPU_THREADS="${MODEL_CPU_THREADS:-4}"
export MODEL_NUM_WORKERS=0
export MODEL_HORIZONS="${MODEL_HORIZONS:-96 192 336}"
export MODEL_SEEDS="${MODEL_SEEDS:-114 514 1919 810 0}"
export MODEL_LOGS_ROOT="${MODEL_LOGS_ROOT:-/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/foundation/logs}"
exec python -u -m foundation.pretrain.launch "$@"
