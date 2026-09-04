# FITS: Modeling Time Series with 10k parameters (Aurora-XPU Adaption)

### Original Work
See the paper/repo from which this has been adapted to ALCF's [Aurora](https://www.alcf.anl.gov/aurora) Supercomputer

- [Code](https://github.com/VEWOXIC/FITS)
- [ArXiV](https://arxiv.org/abs/2307.03756)


### Comments
- FITS is *not foundational*: It is trained from scratch individually for each dataset; not pre-trained on a single pre-train corpus.

### Original repo
This is the official implementation of FITS. Please run the scripts in scripts\FITS for results. Scripts without `_best` are for ablation study and grid search for parameters. Scripts with `_best` are for multiple run on the optimal parameters.

See updates here: [Update](#update)

## Aurora experiment-parallel benchmarks

`scripts/launch_benchmark.sh` is the canonical benchmark interface. The first
argument names the model lane, making the same command usable for FITS and for
future DIFFS, ANG, and ROUTER implementations. Each dataset/horizon/seed
configuration is an independent experiment assigned to one XPU tile.

Train the complete FITS matrix from an interactive Aurora compute node:

```bash
MODEL_MAX_PARALLEL=6 \
  bash scripts/launch_benchmark.sh FITS train \
  ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic
```

The launcher requests all four forecast horizons and the five reporting seeds.
It skips configurations with successful completion markers. An interrupted
configuration is run again from epoch one; optimizer/epoch-level checkpoint
resume is not implemented, and a two-stage B+F/F configuration restarts both
stages.

After training completes, evaluate the checkpoints with the same model-explicit
interface:

```bash
MODEL_MAX_PARALLEL=6 \
  bash scripts/launch_benchmark.sh FITS test \
  ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic
```

Then generate the shared model metrics and five-seed summary tables:

```bash
python scripts/merge_model_metrics.py
python scripts/summarize_model_metrics.py
```

The pool uses no `mpiexec` or model-level DDP. It sets a distinct flat
`ZE_AFFINITY_MASK` in every child, binds four CPU cores per child, staggers
starts by two seconds, and defaults to zero DataLoader subprocesses per
experiment. Tune the operational concurrency without changing experiment
hyperparameters:

```bash
MODEL_MAX_PARALLEL=6 MODEL_LAUNCH_STAGGER_SECONDS=3 \
  bash scripts/launch_benchmark.sh FITS train ETTh1 ETTh2
```

Scheduler evidence is written below
`logs/FITS/aurora/benchmark_runs/<run-id>/`: `manifest.json`, `attempts.csv`,
`events.jsonl`, `summary.json`, per-attempt logs, XPU health snapshots, raw
`xpu-smi` telemetry, and a telemetry peak/counter summary. A Level Zero failure
quarantines its tile and retries the task once elsewhere. An unhandled signal
or critical health report stops new admissions and drains already-running
experiments.

Aurora `ze_peak` safety is primarily a teardown concern. On SIGINT/SIGTERM the
pool starts no new work and sends no force signals. FITS records the signal,
stops between batches, synchronizes submitted XPU work, releases references,
and exits through normal Python teardown. Do not use `qdel -W force`; allow at
least 60 seconds of scheduler kill grace. Monitoring is useful evidence but
cannot by itself prove that the next PBS prologue will pass.

See [the experiment-pool runbook](docs/aurora_experiment_pool.md) for resume,
failure, and tuning controls.

Once another model lane is implemented, substitute its model name without
changing the experiment-pool workflow, for example
`scripts/launch_benchmark.sh DIFFS train ...`. The model-specific lane supplies
its architecture and hyperparameters; the shared launcher supplies scheduling,
XPU isolation, health tracking, and lifecycle records.

## Aurora multi-tile training

`run_longExp_F.py` supports single-node distributed training with one XCCL/DDP
process per PyTorch-visible XPU tile. Aurora's default FLAT hierarchy exposes 12
tiles per node:

```bash
module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset ZE_AFFINITY_MASK
export OMP_NUM_THREADS=4
export MASTER_ADDR="$(hostname | cut -d. -f1).hsn.cm.aurora.alcf.anl.gov"
export MASTER_PORT=29500
export CPU_BIND="verbose,list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"

mpiexec --pmi=pmix --envall -n 12 --ppn 12 --cpu-bind=${CPU_BIND} \
  python -u run_longExp_F.py --distributed --is_training 1 ...
```

Distributed mode currently supports `FITS` and `Real_FITS`, is training-only,
and deliberately rejects test or prediction execution. Evaluate the saved
rank-zero checkpoint later with an ordinary `--is_training 0` invocation.

The runner's existing augmentation rules are applied to `--batch_size` first;
the resulting value is the DDP **global** batch and must be divisible by the
world size. The runner prints both global and per-rank values before training.
`--ddp_num_workers` controls loader workers per rank and defaults to zero to
avoid creating 120 workers on one node.

Do not set a job-wide `ZE_AFFINITY_MASK` for this mode. Rank placement comes
from `PALS_LOCAL_RANKID`, and ambiguous or incompatible launch configurations
fail before model construction.

For a single-XPU interactive-node smoke test, selected-result configurations,
and separate checkpoint evaluation, see
[scripts/FITS/aurora/README.md](scripts/FITS/aurora/README.md). The historical
author ablation scripts are preserved separately in
[`scripts/FITS/legacy_author_ablation_scripts/`](scripts/FITS/legacy_author_ablation_scripts/).

## Analysis

The discovered bug predominantly impacts results on smaller datasets like ETTh1 and ETTh2. Interestingly, for other datasets, certain models, such as PatchTST on ETTm1, demonstrate enhanced performance. FITS still maintains its good enough and comparable-to-sota performance.

## Datasets

The experiment runners default to the shared FITS data directory:

```text
/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/FITS_data/
```

`run_longExp_F.py` stores FITS checkpoints on Lustre by default:

```text
/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/FITS_checkpoints/FITS/
```

FITS test arrays, plots, and mergeable metrics are stored separately under:

```text
/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/FITS_results/FITS/
```

The model-lane convention for FITS and future DIFFS, ANG, and ROUTER work is
documented in [docs/model_lanes.md](docs/model_lanes.md).

Download the four ETT datasets there:

```bash
fits_data_dir="/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/FITS_data"
mkdir -p "$fits_data_dir"

for name in ETTh1 ETTh2 ETTm1 ETTm2; do
  curl -fL --retry 3 \
    -o "$fits_data_dir/${name}.csv" \
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/${name}.csv"
done
```

Download the remaining FITS forecasting benchmarks:

```bash
curl -fL --retry 3 \
  -o "$fits_data_dir/weather.csv" \
  'https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv?download=true'

curl -fL --retry 3 \
  -o "$fits_data_dir/electricity.csv" \
  'https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv?download=true'

curl -fL --retry 3 \
  -o "$fits_data_dir/traffic.csv" \
  'https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv?download=true'
```

The FITS scripts use this directory by default. Override it for another data
location with `--root_path /path/to/data/`.
