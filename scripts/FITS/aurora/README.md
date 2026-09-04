# Aurora single-XPU FITS runs

These scripts run one Python process on the first XPU visible to PyTorch.  They
source the repository's Aurora virtual-environment activation script, use the
shared default data directory, and do not assign a tile themselves.  A batch
scheduler's `ZE_AFFINITY_MASK` is preserved.  On an interactive node, set
`FITS_ZE_AFFINITY_MASK` only if you need to select a particular visible tile.
Checkpoints are stored on Lustre under
`/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/FITS_checkpoints/FITS/`.
Override that location per invocation with `FITS_CHECKPOINTS_ROOT=/path`.

## First training test

From the repository root:

```bash
bash scripts/FITS/aurora/smoke_etth1.sh
```

This performs two epochs of the selected ETTh1-96 configuration: FITS,
multivariate inputs, look-back 360, prediction horizon 96, frequency order 6,
forecast-only loss, Adam learning rate 5e-4, and seed 514.  Its only deliberate
shortcut is the two epochs and passed batch size 8 (the runner turns that into
an effective batch size of 16).  It trains only; it does not touch test data.

Evaluate that saved checkpoint later with:

```bash
bash scripts/FITS/aurora/evaluate_etth1_smoke.sh
```

The test command writes its MSE to the terminal, `result.txt`, and a per-run
log under `logs/FITS/aurora/`; it also saves arrays under `results/` and plots
under `test_results/`.

## Selected-result runs

`run_selected.sh` accepts:

```bash
bash scripts/FITS/aurora/run_selected.sh train DATASET HORIZON SEED
bash scripts/FITS/aurora/run_selected.sh test  DATASET HORIZON SEED
```

For example, one full ETTh1 run and its separate evaluation are:

```bash
bash scripts/FITS/aurora/run_selected.sh train ETTh1 96 514
bash scripts/FITS/aurora/run_selected.sh test  ETTh1 96 514
```

The default is the 100-epoch cap, four loader workers, and the learning rate,
patience, effective batch size, look-back, cutoff construction, loss mode, and
channel count recorded in the authors' updated scripts/logs.  Early stopping
can end a run earlier.  Override operational values without editing a script:

```bash
FITS_TRAIN_EPOCHS=50 FITS_NUM_WORKERS=0 \
  bash scripts/FITS/aurora/run_selected.sh train ETTh1 96 514
```

The selected configurations are:

| Dataset | Look-back | Base period | Frequency order | Loss mode |
| --- | ---: | ---: | ---: | --- |
| ETTh1 | 360 | 24 | 6 | forecast only |
| ETTh2 | 720 | 24 | 6 | B+F then forecast at 96; forecast only otherwise |
| ETTm1 | 360 at 96/192; 720 at 336/720 | 96 | 14 | forecast only |
| ETTm2 | 720 | 96 | 14 | B+F then forecast |
| Weather | 720 | 144 | 12/12/8/12 | B+F, F, B+F, F by horizon |
| Electricity | 720 | 24 | 10 | B+F, B+F, F, B+F by horizon |
| Traffic | 720 | 24 | 10 | F, F, B+F, B+F by horizon |

Here `F` is forecast-only (`--train_mode 1`); `B+F` is backcast-plus-forecast
pretraining followed by forecast-only fine tuning (`--train_mode 2`).  Weather
uses the authors' per-channel (`--individual`) variant.  The scripts calculate
the cutoff exactly as the runner does from the look-back, base period, and
frequency order; they do not hard-code a separate cutoff.

To follow the paper's five-run reporting, run every desired dataset/horizon
with seeds `114 514 1919 810 0`, then evaluate those five checkpoints with the
same commands and aggregate their five reported MSE values.  Each model ID
contains the seed, preventing checkpoint/result collisions present in the
original shell scripts.

## Reproducibility boundary

This is the closest executable protocol available from the paper and updated
repository, not a guarantee of bitwise paper reproduction.  The paper does not
specify seeds or optimization details; the authors' logs supply the Adam
learning rate, patience, 100-epoch cap, and batch size.  Their old logs were
produced on CUDA and also evaluated during training.  This repository now has
the corrected no-`drop_last` test loader and performs test evaluation separately,
while the Aurora XPU/PyTorch implementation can have different numerical FFT
and complex-linear behavior.  Therefore compare five-run mean/std rather than
requiring a particular individual seed to equal a published three-decimal MSE.

The original `scripts/FITS/*.sh` remain useful for the full validation-grid
ablations.  Do not run them unchanged here: they override `--root_path` with
`./dataset/`, contain fixed tile masks, and reuse model IDs across seeds.
