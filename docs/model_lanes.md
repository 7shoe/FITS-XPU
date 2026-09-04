# Model lanes

The repository separates research models by model name while retaining shared
forecasting infrastructure.

```text
models/<MODEL>/                 implementation package
scripts/<MODEL>/aurora/         Aurora launchers
scripts/<MODEL>/legacy_*/       non-supported historical scripts, if any
/lus/.../TimeSeriesTraining/
  <MODEL>_checkpoints/<MODEL>/  checkpoints
  <MODEL>_results/<MODEL>/      per-setting arrays, plots, metrics.csv
```

`FITS` is active. `DIFFS`, `ANG`, and `ROUTER` are reserved lanes; they contain
no runnable model or launcher yet. Implement each with a `Model` export in its
package and an Aurora `run_selected.sh` accepting the standard command form:

```bash
bash scripts/<MODEL>/aurora/run_selected.sh {train|test} DATASET HORIZON SEED
```

Use the common dispatcher for an individual implemented model:

```bash
bash scripts/launch_model.sh FITS train ETTh1 96 514
```

Use `all` to run every model that has an implemented `run_selected.sh`,
sequentially, with the same arguments:

```bash
bash scripts/launch_model.sh all test ETTh1 96 514
```

Run every standard horizon and the five reporting seeds for one or more
datasets with:

```bash
bash scripts/launch_benchmark.sh FITS train ETTh1
```

This launcher is experiment-parallel: it dynamically assigns independent
dataset/horizon/seed tasks to distinct flat XPU tiles. A future lane's
`run_selected.sh` must honor `MODEL_ZE_AFFINITY_MASK`, require exactly one
visible XPU when `MODEL_EXPECT_SINGLE_XPU=1`, and provide cooperative
SIGINT/SIGTERM teardown. See [the Aurora pool contract](aurora_experiment_pool.md).

FITS evaluation writes a standard `metrics.csv`; merge all model lanes after
evaluation and produce a five-seed comparison table with:

```bash
python scripts/merge_model_metrics.py
python scripts/summarize_model_metrics.py
```
