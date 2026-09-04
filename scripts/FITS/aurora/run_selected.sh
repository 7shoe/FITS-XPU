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
source /home/siebenschuh/Projects/Aurora_HPC/env_tsfm/activate_tsfm_venv.sh

cd "$repo_root"

# Preserve scheduler-assigned affinity.  Set FITS_ZE_AFFINITY_MASK only when
# manually choosing a visible tile on an interactive node.
if [[ -n "${FITS_ZE_AFFINITY_MASK:-}" ]]; then
    export ZE_AFFINITY_MASK="$FITS_ZE_AFFINITY_MASK"
fi
export OMP_NUM_THREADS="${FITS_OMP_NUM_THREADS:-4}"
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

python - <<'PY'
import torch
if not hasattr(torch, "xpu") or not torch.xpu.is_available():
    raise SystemExit("No PyTorch XPU is visible; request/allocate a compute node first.")
print(f"Using PyTorch {torch.__version__}; visible XPUs: {torch.xpu.device_count()}")
PY

# The original runner doubles this passed value when no augmentation is used.
# Thus 64 becomes the historical effective batch size of 128 (Weather: 32->64).
epochs="${FITS_TRAIN_EPOCHS:-100}"
workers="${FITS_NUM_WORKERS:-4}"
model_id="aurora_${dataset}_sl${seq_len}_pl${pred_len}_h${h_order}_m${train_mode}_s${seed}"
log_dir="logs/FITS/aurora"
mkdir -p "$log_dir"

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

if [[ "$action" == train ]]; then
    # Training intentionally has no --run_test.  This preserves a held-out test
    # set for the separate evaluation step below.
    python -u run_longExp_F.py "${args[@]}" |& tee "$log_dir/${model_id}.train.log"
else
    args[1]=0
    python -u run_longExp_F.py "${args[@]}" |& tee "$log_dir/${model_id}.test.log"
fi
