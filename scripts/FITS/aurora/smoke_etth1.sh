#!/usr/bin/env bash
# A short, single-XPU functional test of the ETTh1-96 training path.

set -e -o pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FITS_TRAIN_EPOCHS=2 FITS_NUM_WORKERS=0 \
    bash "$script_dir/run_selected.sh" train ETTh1 96 514
