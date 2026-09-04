#!/usr/bin/env bash
# Dispatch a standard single-model command to an implemented model lane.

set -e -o pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 {FITS|DIFFS|ANG|ROUTER|all} {train|test} [MODEL-ARGUMENTS...]" >&2
    exit 2
fi

requested_model="$1"
shift
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_model() {
    local model="$1"
    shift
    local runner="scripts/${model}/aurora/run_selected.sh"
    if [[ ! -f "$runner" ]]; then
        echo "${model} has a reserved lane but no implemented Aurora launcher: ${runner}" >&2
        return 2
    fi
    bash "$runner" "$@"
}

if [[ "$requested_model" == all ]]; then
    found=0
    for runner in scripts/*/aurora/run_selected.sh; do
        [[ -f "$runner" ]] || continue
        model="$(basename "$(dirname "$(dirname "$runner")")")"
        found=1
        bash "$runner" "$@"
    done
    if [[ "$found" == 0 ]]; then
        echo "No implemented model launchers found." >&2
        exit 2
    fi
else
    run_model "$requested_model" "$@"
fi
