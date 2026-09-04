#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK="${MODEL_ZE_AFFINITY_MASK:?pool must assign a tile}"
export OMP_NUM_THREADS="${MODEL_CPU_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
exec python -u -m foundation.pretrain.train --model "$1" --corpus "$2" --horizon "$3" --seed "$4"
