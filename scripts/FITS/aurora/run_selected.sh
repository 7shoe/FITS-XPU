#!/usr/bin/env bash
# Train or evaluate one of the reported FITS forecasting configurations on one
# Aurora XPU tile.  The model ID includes the seed so five-run experiments do
# not overwrite one another's checkpoint.

set -e -o pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 {train|test} {ETTh1|ETTh2|ETTm1|ETTm2|weather|electricity|traffic} {96|192|336|720} SEED" >&2
    exit 2
fi

action="$1"
dataset="$2"
pred_len="$3"
seed="$4"

case "$action" in
    train|test) ;;
    *) echo "action must be train or test" >&2; exit 2 ;;
esac

case "$pred_len" in
    96|192|336|720) ;;
    *) echo "prediction length must be 96, 192, 336, or 720" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ "${MODEL_ENV_READY:-0}" != 1 ]]; then
    source /home/siebenschuh/Projects/Aurora_HPC/env_tsfm/activate_tsfm_venv.sh
fi

cd "$repo_root"

# Preserve scheduler-assigned affinity. MODEL_ZE_AFFINITY_MASK is the shared
# experiment-pool contract; FITS_ZE_AFFINITY_MASK remains a compatible manual
# override for direct runs.
selected_tile="${MODEL_ZE_AFFINITY_MASK:-${FITS_ZE_AFFINITY_MASK:-}}"
if [[ -n "$selected_tile" ]]; then
    export ZE_AFFINITY_MASK="$selected_tile"
fi
cpu_threads="${FITS_OMP_NUM_THREADS:-${MODEL_CPU_THREADS:-4}}"
export OMP_NUM_THREADS="$cpu_threads"
export MKL_NUM_THREADS="$cpu_threads"
export OPENBLAS_NUM_THREADS="$cpu_threads"
export NUMEXPR_MAX_THREADS="$cpu_threads"
export NUMEXPR_NUM_THREADS="$cpu_threads"
export MODEL_REQUIRE_XPU=1
checkpoints_root="${FITS_CHECKPOINTS_ROOT:-/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/FITS_checkpoints/FITS}"
results_root="${FITS_RESULTS_ROOT:-/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/FITS_results/FITS}"

case "$dataset" in
    ETTh1)
        data=ETTh1; data_path=ETTh1.csv; channels=7; seq_len=360
        base_t=24; h_order=6; train_mode=1; passed_batch=64; patience=20
        ;;
    ETTh2)
        data=ETTh2; data_path=ETTh2.csv; channels=7; seq_len=720
        base_t=24; h_order=6; passed_batch=64; patience=20
        if [[ "$pred_len" == 96 ]]; then train_mode=2; else train_mode=1; fi
        ;;
    ETTm1)
        data=ETTm1; data_path=ETTm1.csv; channels=7; base_t=96
        h_order=14; train_mode=1; passed_batch=64; patience=20
        if [[ "$pred_len" == 96 || "$pred_len" == 192 ]]; then seq_len=360; else seq_len=720; fi
        ;;
    ETTm2)
        data=ETTm2; data_path=ETTm2.csv; channels=7; seq_len=720
        base_t=96; h_order=14; train_mode=2; passed_batch=64; patience=20
        ;;
    weather)
        data=custom; data_path=weather.csv; channels=21; seq_len=720
        base_t=144; passed_batch=32; patience=3; individual=(--individual)
        case "$pred_len" in
            96)  h_order=12; train_mode=2 ;;
            192) h_order=12; train_mode=1 ;;
            336) h_order=8;  train_mode=2 ;;
            720) h_order=12; train_mode=1 ;;
        esac
        ;;
    electricity)
        data=custom; data_path=electricity.csv; channels=321; seq_len=720
        base_t=24; h_order=10; passed_batch=64; patience=20
        case "$pred_len" in
            96|192|720) train_mode=2 ;;
            336)        train_mode=1 ;;
        esac
        ;;
    traffic)
        data=custom; data_path=traffic.csv; channels=862; seq_len=720
        base_t=24; h_order=10; passed_batch=64; patience=10
        case "$pred_len" in
            96|192) train_mode=1 ;;
            336|720) train_mode=2 ;;
        esac
        ;;
    *) echo "unknown dataset: $dataset" >&2; exit 2 ;;
esac

# The original runner doubles this passed value when no augmentation is used.
# Thus 64 becomes the historical effective batch size of 128 (Weather: 32->64).
epochs="${FITS_TRAIN_EPOCHS:-100}"
workers="${FITS_NUM_WORKERS:-${MODEL_NUM_WORKERS:-4}}"
model_id="aurora_${dataset}_sl${seq_len}_pl${pred_len}_h${h_order}_m${train_mode}_s${seed}"
logs_root="${MODEL_LOGS_ROOT:-$repo_root/logs}"
if [[ "$logs_root" != /* ]]; then
    logs_root="$repo_root/$logs_root"
fi
log_dir="$logs_root/FITS/aurora"
mkdir -p "$log_dir"
setting="${model_id}_FITS_${data}_ftM_sl${seq_len}_ll48_pl${pred_len}_H${h_order}_0"
status_dir="$log_dir/run_status"
status_file="$status_dir/${model_id}.${action}.complete"
mkdir -p "$status_dir"

# Refuse duplicate pools/direct launches before either process can write the
# same checkpoint or result setting. The descriptor remains inherited by the
# Python process, so the lock also survives an accidental launcher-shell exit.
exec 9>"$status_dir/${model_id}.lock"
if ! flock -n 9; then
    echo "Run is already active: ${model_id} ${action}" >&2
    exit 75
fi

if [[ "${MODEL_SKIP_COMPLETED:-0}" == 1 && -f "$status_file" ]]; then
    echo "Already completed; skipping ${model_id} ${action}"
    exit 0
fi

args=(
    --is_training 1
    --model_id "$model_id"
    --model FITS
    --data "$data"
    --data_path "$data_path"
    --checkpoints "$checkpoints_root"
    --results_root "$results_root"
    --features M
    --seq_len "$seq_len"
    --label_len 48
    --pred_len "$pred_len"
    --enc_in "$channels"
    --train_mode "$train_mode"
    --H_order "$h_order"
    --base_T "$base_t"
    --seed "$seed"
    --itr 1
    --batch_size "$passed_batch"
    --learning_rate 0.0005
    --patience "$patience"
    --train_epochs "$epochs"
    --num_workers "$workers"
    --use_gpu true
)

if [[ ${#individual[@]} -gt 0 ]]; then
    args+=("${individual[@]}")
fi

received_signal=""
python_pid=""

forward_graceful_signal() {
    local signal_name="$1"
    received_signal="$signal_name"
    echo "run_selected received ${signal_name}; requesting cooperative Python shutdown" >&2
    if [[ -n "$python_pid" ]] && kill -0 "$python_pid" 2>/dev/null; then
        kill -s "$signal_name" "$python_pid" 2>/dev/null || true
    fi
}

trap 'forward_graceful_signal TERM' TERM
trap 'forward_graceful_signal INT' INT

run_and_log() {
    local log_path="$1"
    shift
    local rc

    set +e
    # tee ignores job-control signals and drains Python's output through EOF;
    # otherwise a qdel/Control-C could kill tee first and give Python a broken
    # stdout pipe while it is trying to shut its XPU context down cleanly.
    python -u run_longExp_F.py "$@" > >(trap '' INT TERM; tee "$log_path") 2>&1 &
    python_pid=$!
    while true; do
        wait "$python_pid"
        rc=$?
        if ! kill -0 "$python_pid" 2>/dev/null; then
            break
        fi
    done
    # Let the process-substitution tee consume EOF before recording lifecycle.
    wait 2>/dev/null || true
    set -e

    echo "RUN_LIFECYCLE model_id=${model_id} action=${action} "\
"python_exit=${rc} signal_observed=${received_signal:-none} forced_signal=none"
    return "$rc"
}

run_rc=0
if [[ "$action" == train ]]; then
    # Training intentionally has no --run_test. This preserves a held-out test
    # set for the separate evaluation step below.
    run_and_log "$log_dir/${model_id}.train.log" "${args[@]}" || run_rc=$?
else
    args[1]=0
    run_and_log "$log_dir/${model_id}.test.log" "${args[@]}" || run_rc=$?
fi

if [[ "$run_rc" -eq 0 && -z "$received_signal" ]]; then
    status_tmp="${status_file}.tmp.$$"
    printf 'setting=%s\naction=%s\ntile=%s\ncompleted_utc=%s\n' \
        "$setting" "$action" "${selected_tile:-unbound}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_tmp"
    mv "$status_tmp" "$status_file"
fi

exit "$run_rc"
