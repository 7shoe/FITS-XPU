#!/usr/bin/env bash
# Evaluate the checkpoint made by smoke_etth1.sh, without further training.

set -e -o pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FITS_NUM_WORKERS=0 bash "$script_dir/run_selected.sh" test ETTh1 96 514
