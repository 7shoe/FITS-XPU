# Corpus pretraining on Aurora

The loader follows [the supplied integration specification](../../pretraining_integration_v1_v2_v3.md)
for corpus discovery, split membership, family weights, and budget subsets.
The forecasting objective and optimization settings below are new choices for
this repository, not settings reported in the FITS paper.

## Launch FITS on v1

From the repository on an allocated interactive compute node:

```bash
MODEL_MAX_PARALLEL=6 \
MODEL_SEEDS="114 514 1919 810 0" \
MODEL_HORIZONS="96 192 336" \
PRETRAIN_CONTEXT=512 \
PRETRAIN_BUDGET_OBS=1e9 \
PRETRAIN_STEPS=10000 \
PRETRAIN_BATCH_SIZE=256 \
bash scripts/launch_pretrain.sh FITS train v1
```

This schedules 15 independent experiments (five seeds × three horizons), with
up to six active single-tile workers. Omit `MODEL_MAX_PARALLEL` to use the pool's
detected-tile default. Adding `v2 v3` schedules 45 experiments; versions are
**separate runs**, not concatenated corpora or sequential continuation stages.
Multiple implemented models can precede `train`; models are processed in order,
with parallel experiments within each model. Unsupported models fail up front.
`launch_pretraining.sh` is an alias. No distributed gradient training is used.

For a first XPU smoke test, use the same launcher with smaller settings:

```bash
MODEL_MAX_PARALLEL=1 MODEL_SEEDS=114 MODEL_HORIZONS=96 \
PRETRAIN_CONTEXT=512 PRETRAIN_BUDGET_OBS=1048576 \
PRETRAIN_STEPS=10 PRETRAIN_BATCH_SIZE=8 PRETRAIN_VAL_WINDOWS=16 \
bash scripts/launch_pretrain.sh FITS train v1
```

The launcher sources the existing TSFM activation script and installs nothing.
It uses the hardware-provided torch plus existing NumPy and PyArrow. Do not
replace torch/transformers or install the original FITS requirements wholesale.
The pool assigns flat tiles, CPU affinity, startup staggering, health telemetry,
and existing Level Zero retry/quarantine behavior. This reduces risk but cannot
guarantee prevention of driver failures. Avoid forced process termination.

## Data protocol

- v1: `msm/tier1/MANIFEST.json`, only `gifteval_uni` and `kernelsynth`, regular
  univariate 1024-slot rows. Other manifest exports are recorded as exclusions.
- v2: `msm/enhanced_pretrain_v2/wave1/MANIFEST_10B.json`;
  validation comes from `enhanced_pretrain_v2/holdout_25M/val`.
- v3: `msm/extensive_pretrain_v3/MANIFEST_V3.json`;
  validation comes from `extensive_pretrain_v3/holdout_25M/val`.

These paths are under `/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesData/`.
No download is needed. Overrides are `PRETRAIN_V1_MANIFEST`,
`PRETRAIN_V1_SOURCE_ROOT`, `PRETRAIN_V1_HOLDOUT_ROOT` (likewise V2/V3).

The loader memory-maps float32 values and reads actual parquet offsets/lengths.
Only `train` rows are sampled, never `val`, `test`, or `excluded`. It verifies
geometry, bounds, and split counts; it does not rerun the upstream corpus's
parent-group or value-level decontamination audit. v3 and v2 holdouts must not
be assumed mutually disjoint. No test export is loaded by this trainer.

`floor(PRETRAIN_BUDGET_OBS/1024)` is a **stored-slot budget**, including v3
padding, not the number of finite observations or optimizer exposures. Per-source
caps are proportional to train-row counts, rounded down. Seeded permutation
prefixes provide nested subsets as budgets grow. v1 samples sources proportional
to train counts; v2/v3 reconstruct family-share × within-family series-count
weights and renormalize after budget exclusions. Sampling is with replacement.

Train crops are random contiguous context+horizon slices within a stored row.
Validation uses deterministic tail crops from fixed curated validation subsets:
up to 1024 rows per source by default (two sources in v1, one holdout in v2/v3).
This is subset validation, not exhaustive benchmark evaluation.

Finite context values determine per-window mean/sample variance. Missing input
values become zero after normalization; missing future targets are masked from
MSE. Rows without two finite context observations or any finite target, or with
extreme normalized magnitude, are skipped and counted. There is no clipping.
FITS still applies its internal instance normalization; its architecture is not
mask-aware. Its shared-channel complex layer predicts all future values directly.
Training uses forecast-only masked MSE, Adam at constant learning rate 0.0005,
full float32/complex64 precision, and no augmentation or early stopping.

## Configuration and storage

Additional defaults:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PRETRAIN_CUT_FREQ` | 64 | Retained FITS spectral bins; not an inferred harmonic |
| `PRETRAIN_LR` | 0.0005 | Constant Adam learning rate |
| `PRETRAIN_SUBSET_SEED` | 1234 | Fixed budget/validation subset selection |
| `PRETRAIN_SAVE_EVERY` | 100 | Full-state checkpoint interval in updates |
| `PRETRAIN_VALIDATE_EVERY` | 500 | Curated validation interval; also runs at final step |
| `PRETRAIN_VAL_WINDOWS` | 1024 | Maximum validation rows per source |
| `PRETRAIN_LOG_EVERY` | 10 | Training progress interval |

Context+horizon must fit in 1024 slots and be even for the current FITS inverse
FFT. One context/horizon/cutoff defines one architecture geometry. In particular,
context 512 cannot be combined with horizon 720. Transfer to different geometry
requires an explicit weight-transfer rule, not yet implemented.

Default checkpoints:

```text
/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/FITS_checkpoints/
  FITS/                         # existing dataset-specific runs, untouched
  pretrained/
    FITS/pretrain_v1_sl512_pl96_cf64_obs1000000000_steps10000_s114_<hash>/
      checkpoint.pth            # best validation model weights, for future adaptation
      last.pth                  # optimizer, weights, step, RNG states, source statistics
      config.json               # objective, geometry, optimization, code hashes
      corpus.json               # manifest/index fingerprints, weights, realized caps
      training.jsonl            # training and validation records
      source_stats.json         # source exposure and invalid-row counters
      complete.json             # written only after all requested steps finish
    _index_cache/               # shared deterministic subset indexes
  adapt/<MODEL>/                # reserved downstream checkpoints
```

`PRETRAIN_CHECKPOINT_ROOT` overrides the `pretrained` directory, not its parent.
`PRETRAIN_INDEX_CACHE` independently overrides the shared cache. Future model
checkpoints use the same naming convention under `pretrained/DIFFS`, etc.

Pool manifests, status, health telemetry and per-attempt console logs go under:

```text
/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/foundation/logs/
  FITS/aurora/pretrain_runs/<launch-id>/
```

Override with `MODEL_LOGS_ROOT`. The first launch scans and hashes parquet indexes
and creates locked caches; this may take substantial time before GPU work starts.
Index preparation holds one source's offsets/permutation in memory at a time;
it does not load all corpus values, but large sources still require RAM.

Rerun the **same command** to resume incomplete runs at their last saved update,
including Adam and RNG state, or skip completed runs. SIGINT/SIGTERM requests a
safe-boundary save and drain. A crash can replay updates since the last snapshot.
Configuration/corpus/code changes select a new hashed directory, not silent
continuation. Changing `PRETRAIN_STEPS` likewise creates a new experiment.
Only load trusted local full-state checkpoints. Bitwise cross-device equivalence
is not promised. A lock prevents concurrent writers to the same experiment.

## Verification

```bash
python -m unittest foundation.pretrain.test_integration -v
```

Tests cover stored-row boundaries, budget nesting, masking, weight reconstruction,
and CPU interruption/resume equivalence. A two-update CPU smoke test also passed
against the mounted v1 corpus. Production XPU and full v2/v3 training require
compute-node validation. Adaptation and downstream comparison tables are not
implemented in this lane yet.
