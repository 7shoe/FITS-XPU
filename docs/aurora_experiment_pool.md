# Aurora experiment pool

FITS and the planned small-model lanes use experiment-level parallelism: each
dataset/horizon/seed task is an ordinary single-process run bound to one XPU
tile. This preserves the paper's per-run batch size and optimizer behavior,
unlike splitting one tiny model across 12 DDP ranks.

## Run

From an interactive Aurora compute-node allocation:

```bash
unset ZE_AFFINITY_MASK
bash scripts/launch_benchmark.sh FITS train \
  ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic
```

For FITS, the complete isolated train/evaluate/summarize workflow is:

```bash
bash scripts/FITS/aurora/run_full_reproduction.sh
```

It is resumable and keeps its checkpoints, results, logs, and summary beneath
the tagged reproduction directory on Lustre. Set `FITS_REPRO_TAG` to retain
multiple independent reproductions.

The launcher sources the repository's TSFM environment, selects the FLAT
device hierarchy, uses `xpu-smi` to map Aurora's six cards to 12 tiles, and
uses all of them by default. The activated environment is inherited rather
than re-created by all children. It refuses a pre-existing
`ZE_AFFINITY_MASK`, because a pool must own the per-child masks. Each pooled
child verifies that PyTorch sees exactly one device, which is locally named
`xpu:0` even when its physical flat tile is 7.

The four horizons and five paper-reporting seeds create 20 tasks per dataset.
Restrict either dimension for a smoke or resumed subset:

```bash
MODEL_HORIZONS='96 192' MODEL_SEEDS='114 514' \
  bash scripts/launch_benchmark.sh FITS train ETTh1
```

Parallel evaluation uses the same scheduler:

```bash
bash scripts/launch_benchmark.sh FITS test \
  ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic
python scripts/merge_model_metrics.py
python scripts/summarize_model_metrics.py
```

Evaluation writes one `metrics.csv` per setting, so no two processes append to
the same file. The merge step discovers both these files and older lane-level
metrics and removes exact duplicate rows.

## Controls

| Variable | Default | Meaning |
| --- | ---: | --- |
| `MODEL_MAX_PARALLEL` | detected XPU count | Maximum simultaneous experiments; one per flat device by default |
| `MODEL_XPU_TILES` | detected `0..N-1` | Comma/space list or range such as `0-5` |
| `MODEL_LAUNCH_STAGGER_SECONDS` | `2` | Delay between XPU process starts |
| `MODEL_NUM_WORKERS` | `0` | DataLoader subprocesses per experiment |
| `MODEL_CPU_THREADS` | `4` | CPU math-library threads per experiment |
| `MODEL_ZE_RETRIES` | `1` | Retries on a different tile after a Level Zero failure |
| `MODEL_SKIP_COMPLETED` | `1` | Skip runs with a successful completion marker |
| `MODEL_ENABLE_XPU_SMI` | `1` | Capture health and telemetry |
| `MODEL_HEALTH_INTERVAL_SECONDS` | `30` | Health-poll interval; `0` disables polling |
| `MODEL_TELEMETRY_INTERVAL_SECONDS` | `5` | Raw telemetry sampling interval |
| `MODEL_CPU_BIND` | `1` | Use the established four-core Aurora tile bindings |
| `MODEL_CPU_SETS` | Aurora mapping | Colon-separated custom CPU sets |

Use `MODEL_MAX_PARALLEL=1` for the former sequential behavior. Set
`MODEL_SKIP_COMPLETED=0` only when intentionally replacing a successful run.
Completion markers were introduced with the pool; old checkpoints without a
marker are conservatively retrained rather than guessed to be complete.

## Evidence and recovery

Every invocation creates
`logs/<MODEL>/aurora/benchmark_runs/<run-id>/` containing:

- `manifest.json`: requested tasks, placement, concurrency, and policy;
- `attempts.csv`: tile, PID, timestamps, exit status, classification, retry;
- `events.jsonl`: incremental lifecycle and health events;
- `summary.json`: final task counts, signal state, and quarantined tiles;
- `attempt_logs/`: immutable output for every attempt, including retries;
- `xpu_health_start.txt` / `xpu_health_end.txt` and periodic snapshots;
- `xpu_smi_telemetry.csv` and `xpu_smi_summary.json`.

The scheduler returns nonzero if any task fails, is not started, the scheduler
receives a signal, a tile is quarantined, or critical health is observed. A
retry may finish all requested tasks, but a device incident still gets a
non-green safety verdict. Re-run the same command after resolving the cause;
successful pooled runs are skipped by their completion markers.
Per-setting file locks also refuse duplicate train/test processes so two pools
cannot concurrently modify the same checkpoint or result directory.

Because `ze_peak` belongs to the next job's prologue, check the recorded host
later from a login node (and repeat after that host receives another job):

```bash
python scripts/check_aurora_ze_peak.py \
  logs/FITS/aurora/benchmark_runs/<run-id>
```

The checker appends an immutable observation to
`/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/ze_peak_ledger.jsonl`.
A green observation means only “nothing is visible yet”; it is not proof of a
clean scheduler boundary.

## `ze_peak` safety contract

ALCF's `ze_peak` is a PBS prologue check for the next job. The relevant known
failure mode is abrupt termination of Python processes that still own Level
Zero contexts. Utilization or temperature telemetry is not the definition of
this failure.

The pool therefore follows these rules:

1. Validate placement and health before admitting model processes.
2. Never share a tile between active experiments.
3. On SIGINT/SIGTERM, stop admissions and wait for active children; the pool
   never sends TERM or KILL to model processes.
4. FITS handles a signal cooperatively at a batch boundary, drains submitted
   XPU work, and exits through normal interpreter teardown.
5. An unhandled signal is a failed safety verdict: stop admissions and drain.
6. Record every child exit and report `forced_signals_sent: 0` in the summary.

This reduces the application-level exposure identified in the local
AgenticTimer `INTEL_ZE_PEAK_FIX.md` report. It cannot control PBS's final
SIGKILL deadline or prove the subsequent prologue result. Avoid force deletion,
request sufficient walltime, and leave at least 60 seconds for teardown. Future
model lanes must honor `MODEL_ZE_AFFINITY_MASK`, verify one visible XPU, and
implement the same cooperative signal boundary before joining this pool.
