#!/usr/bin/env python3
"""Run independent benchmark experiments on a bounded pool of XPU tiles.

The pool owns scheduling only.  Every experiment remains a normal model-lane
invocation, isolated with one flat ``ZE_AFFINITY_MASK`` value.  On SIGINT or
SIGTERM the pool stops admitting work and waits for active children; it never
force-terminates an initialized XPU process.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path


ZE_FAILURE_PATTERNS = (
    re.compile(r"ZE_RESULT_ERROR_", re.IGNORECASE),
    re.compile(r"UR_RESULT_ERROR_(?:DEVICE_LOST|OUT_OF_RESOURCES|OUT_OF_DEVICE_MEMORY)", re.IGNORECASE),
    re.compile(r"Level[ -]?Zero[^\n]*(?:error|failed|failure)", re.IGNORECASE),
    re.compile(r"device (?:was )?lost", re.IGNORECASE),
    re.compile(r"ze_peak", re.IGNORECASE),
)
GRACEFUL_PATTERN = re.compile(r"GRACEFUL_XPU_SHUTDOWN cleanup complete")
CRITICAL_HEALTH_PATTERN = re.compile(r"\b(?:critical|unhealthy)\b", re.IGNORECASE)
COUNTER_NAMES = (
    "Reset Counter",
    "Programming Errors",
    "Driver Errors",
    "Cache Errors Correctable",
    "Cache Errors Uncorrectable",
    "GPU Memory Errors Correctable",
    "GPU Memory Errors Uncorrectable",
)
ATTEMPT_FIELDS = (
    "run_id", "task_id", "model", "action", "dataset", "horizon", "seed",
    "attempt", "tile", "cpu_set", "pid", "started_utc", "ended_utc",
    "duration_seconds", "exit_code", "status", "failure_class",
    "retry_scheduled", "attempt_log",
)


@dataclass(frozen=True)
class Task:
    task_id: str
    dataset: str
    horizon: int
    seed: int
    attempt: int = 1


@dataclass
class Slot:
    tile: str
    cpu_set: str
    quarantined: bool = False


@dataclass
class ActiveRun:
    task: Task
    slot: Slot
    process: subprocess.Popen
    log_handle: object
    log_path: Path
    started_wall: float
    started_utc: str


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_list(value, cast=str):
    return [cast(item) for item in re.split(r"[\s,]+", value.strip()) if item]


def parse_tiles(value):
    tiles = []
    for token in re.split(r"[\s,]+", value.strip()):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", token)
        if match:
            first, last = map(int, match.groups())
            if last < first:
                raise ValueError("descending tile range: {}".format(token))
            tiles.extend(str(index) for index in range(first, last + 1))
        elif token.isdigit():
            tiles.append(token)
        else:
            raise ValueError("invalid flat XPU tile: {}".format(token))
    if not tiles or len(set(tiles)) != len(tiles):
        raise ValueError("MODEL_XPU_TILES must contain distinct flat tile IDs")
    return tiles


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def event_writer(path):
    def write(event, **payload):
        record = {"time_utc": utc_now(), "event": event, **payload}
        with path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return write


def capture_command(command, output_path, timeout=None):
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        text = completed.stdout or ""
        output_path.write_text(text)
        return completed.returncode, text
    except (OSError, subprocess.TimeoutExpired) as exc:
        text = "{}: {}\n".format(type(exc).__name__, exc)
        output_path.write_text(text)
        return 127, text


def health_is_critical(text):
    return bool(CRITICAL_HEALTH_PATTERN.search(text))


def classify_failure(exit_code, log_path):
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        text = ""
    if exit_code in (130, 143) and GRACEFUL_PATTERN.search(text):
        return "graceful_signal"
    if exit_code < 0:
        return "signal_{}".format(-exit_code)
    # Shells encode a child terminated by signal N as 128 + N. Keep this
    # ahead of log-pattern matching: a cleanup receipt printed before an
    # interpreter segfault is not proof of graceful process exit.
    if 129 <= exit_code <= 192:
        return "signal_{}".format(exit_code - 128)
    if any(pattern.search(text) for pattern in ZE_FAILURE_PATTERNS):
        return "level_zero"
    return "process"


def choose_cpu_sets(tiles):
    configured = os.environ.get("MODEL_CPU_SETS")
    if configured:
        cpu_sets = [item.strip() for item in configured.split(":") if item.strip()]
        if len(cpu_sets) < len(tiles):
            raise ValueError("MODEL_CPU_SETS has fewer entries than MODEL_XPU_TILES")
        return cpu_sets[:len(tiles)]

    # Aurora's 12 tile-local four-core bindings, matching the repository's
    # established one-rank-per-tile DDP placement.
    aurora = [
        "4-7", "8-11", "12-15", "16-19", "20-23", "24-27",
        "56-59", "60-63", "64-67", "68-71", "72-75", "76-79",
    ]
    if all(tile.isdigit() and int(tile) < len(aurora) for tile in tiles):
        return [aurora[int(tile)] for tile in tiles]
    return [""] * len(tiles)


def cpuset_allowed(cpu_set):
    if not cpu_set or not hasattr(os, "sched_getaffinity"):
        return False
    allowed = os.sched_getaffinity(0)
    requested = set()
    for piece in cpu_set.split(","):
        if "-" in piece:
            first, last = map(int, piece.split("-", 1))
            requested.update(range(first, last + 1))
        else:
            requested.add(int(piece))
    return requested.issubset(allowed)


def initialize_telemetry(run_dir, enabled):
    executable = shutil.which("xpu-smi")
    if not enabled or executable is None:
        return None
    telemetry_path = run_dir / "xpu_smi_telemetry.csv"
    telemetry_path.write_text("")
    return telemetry_path


def sample_telemetry(executable, telemetry_path):
    if executable is None or telemetry_path is None:
        return 127
    command = [
        executable, "dump", "-d", "-1", "-m",
        "0,1,2,3,4,5,12,13,14,15,16,18,29,30,35",
        "-n", "1", "--date",
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = completed.stdout or ""
        with telemetry_path.open("a") as handle:
            handle.write(output)
        return completed.returncode
    except OSError as exc:
        with telemetry_path.open("a") as handle:
            handle.write("telemetry sample failed: {}\n".format(exc))
        return 127


def summarize_telemetry(source, destination):
    if not source.exists():
        return
    lines = source.read_text(errors="replace").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines)
         if "Timestamp" in line and "DeviceId" in line),
        None,
    )
    if header_index is None:
        atomic_json(destination, {"available": False, "reason": "no CSV header"})
        return
    reader = csv.DictReader(lines[header_index:])
    values = {}
    samples = 0
    for row in reader:
        normalized = {
            key.strip(): value for key, value in row.items() if key is not None
        }
        device = (normalized.get("DeviceId") or "unknown").strip()
        for raw_name, raw_value in normalized.items():
            if raw_name in ("Timestamp", "DeviceId"):
                continue
            match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", raw_value or "")
            if not match:
                continue
            key = (device, raw_name.strip())
            values.setdefault(key, []).append(float(match.group(0)))
        samples += 1
    metrics = []
    for (device, name), series in sorted(values.items()):
        record = {
            "device": device,
            "metric": name,
            "first": series[0],
            "last": series[-1],
            "minimum": min(series),
            "peak": max(series),
            "samples": len(series),
        }
        if any(counter in name for counter in COUNTER_NAMES):
            record["delta"] = series[-1] - series[0]
        metrics.append(record)
    atomic_json(destination, {"available": bool(metrics), "rows": samples, "metrics": metrics})


def write_attempt(writer, handle, record):
    writer.writerow({key: record.get(key, "") for key in ATTEMPT_FIELDS})
    handle.flush()
    os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("action", choices=("train", "test"))
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--horizons", required=True, help="comma-separated horizons")
    parser.add_argument("--seeds", required=True, help="comma-separated seeds")
    parser.add_argument("--task-kind", choices=("benchmark", "pretrain"), default="benchmark")
    args = parser.parse_args()
    args.horizons = parse_list(args.horizons, int)
    args.seeds = parse_list(args.seeds, int)

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    tiles = parse_tiles(os.environ.get("MODEL_XPU_TILES", "0-11"))
    max_parallel = int(os.environ.get("MODEL_MAX_PARALLEL", len(tiles)))
    if max_parallel < 1:
        raise SystemExit("MODEL_MAX_PARALLEL must be positive")
    tiles = tiles[:min(max_parallel, len(tiles))]
    retries = int(os.environ.get("MODEL_ZE_RETRIES", "1"))
    stagger = float(os.environ.get("MODEL_LAUNCH_STAGGER_SECONDS", "2"))
    health_interval = float(os.environ.get("MODEL_HEALTH_INTERVAL_SECONDS", "30"))
    telemetry_interval = int(os.environ.get("MODEL_TELEMETRY_INTERVAL_SECONDS", "5"))
    dry_run = env_bool("MODEL_DRY_RUN")
    monitor_enabled = env_bool("MODEL_ENABLE_XPU_SMI", default=True) and not dry_run
    cpu_binding = env_bool("MODEL_CPU_BIND", default=True)

    cpu_sets = choose_cpu_sets(tiles)
    if not cpu_binding or shutil.which("taskset") is None:
        cpu_sets = [""] * len(tiles)
    else:
        cpu_sets = [cpu_set if cpuset_allowed(cpu_set) else "" for cpu_set in cpu_sets]
    slots = [Slot(tile=tile, cpu_set=cpu_set) for tile, cpu_set in zip(tiles, cpu_sets)]

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = os.environ.get(
        "MODEL_BENCHMARK_RUN_ID",
        "{}_{}_{}_{}".format(timestamp, args.action, socket.gethostname().split(".")[0], os.getpid()),
    )
    logs_root = Path(os.environ.get("MODEL_LOGS_ROOT", str(repo_root / "logs"))).expanduser()
    if not logs_root.is_absolute():
        logs_root = repo_root / logs_root
    run_dir = logs_root / args.model / "aurora" / ("pretrain_runs" if args.task_kind == "pretrain" else "benchmark_runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    attempts_dir = run_dir / "attempt_logs"
    attempts_dir.mkdir()
    write_event = event_writer(run_dir / "events.jsonl")

    tasks = deque()
    for dataset in args.datasets:
        for horizon in args.horizons:
            for seed in args.seeds:
                task_id = "{}_h{}_s{}".format(dataset, horizon, seed)
                tasks.append(Task(task_id, dataset, horizon, seed))

    manifest = {
        "run_id": run_id,
        "task_kind": args.task_kind,
        "created_utc": utc_now(),
        "host": socket.gethostname(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "model": args.model,
        "action": args.action,
        "datasets": args.datasets,
        "horizons": args.horizons,
        "seeds": args.seeds,
        "task_count": len(tasks),
        "tiles": tiles,
        "cpu_sets": cpu_sets,
        "max_parallel": len(slots),
        "launch_stagger_seconds": stagger,
        "level_zero_retries": retries,
        "dataloader_workers_per_experiment": int(os.environ.get("MODEL_NUM_WORKERS", "0")),
        "cpu_threads_per_experiment": int(os.environ.get("MODEL_CPU_THREADS", "4")),
        "skip_completed": env_bool("MODEL_SKIP_COMPLETED", default=True),
        "dry_run": dry_run,
        "command": sys.argv,
    }
    atomic_json(run_dir / "manifest.json", manifest)

    stop_admissions = False
    received_signal = None
    health_critical_observed = False

    def request_stop(signum, _frame):
        nonlocal stop_admissions, received_signal
        stop_admissions = True
        received_signal = received_signal or int(signum)
        print(
            "\nScheduler received signal {}; no new experiments will start. "
            "Active XPU processes will be reaped without force signals.".format(signum),
            flush=True,
        )
        write_event("scheduler_signal", signal=signum, forced_signal="none")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    xpu_smi = shutil.which("xpu-smi")
    if monitor_enabled and xpu_smi:
        rc, health_text = capture_command(
            [xpu_smi, "health", "-l"], run_dir / "xpu_health_start.txt"
        )
        write_event("health_start", exit_code=rc, critical=health_is_critical(health_text))
        if health_is_critical(health_text):
            atomic_json(run_dir / "summary.json", {
                "run_id": run_id,
                "status": "refused_unhealthy",
                "ended_utc": utc_now(),
            })
            raise SystemExit("xpu-smi reported critical/unhealthy hardware; no experiment was started")
    else:
        (run_dir / "xpu_health_start.txt").write_text("xpu-smi monitoring disabled or unavailable\n")

    telemetry_path = initialize_telemetry(run_dir, monitor_enabled)
    print(
        "Benchmark {}: {} tasks, {} parallel XPU tiles {}, logs {}".format(
            run_id, len(tasks), len(slots), ",".join(tiles), run_dir
        ),
        flush=True,
    )
    print(
        "Safety: stagger={}s, loader-workers={}, ZE retries={}, forced signals=disabled".format(
            stagger, manifest["dataloader_workers_per_experiment"], retries
        ),
        flush=True,
    )

    attempts_handle = (run_dir / "attempts.csv").open("w", newline="")
    attempts_writer = csv.DictWriter(attempts_handle, fieldnames=ATTEMPT_FIELDS)
    attempts_writer.writeheader()
    attempts_handle.flush()

    active = {}
    final_status = {}
    quarantined_tiles = set()
    failure_counts = Counter()
    next_launch_time = time.monotonic()
    next_health_time = time.monotonic() + health_interval
    next_telemetry_time = time.monotonic()

    try:
        while tasks or active:
            now = time.monotonic()

            if (monitor_enabled and xpu_smi and telemetry_interval > 0 and
                    now >= next_telemetry_time):
                rc = sample_telemetry(xpu_smi, telemetry_path)
                write_event("telemetry_sample", exit_code=rc)
                next_telemetry_time = time.monotonic() + telemetry_interval

            if (monitor_enabled and xpu_smi and health_interval > 0 and
                    now >= next_health_time and not stop_admissions):
                health_path = run_dir / "xpu_health_{}.txt".format(int(time.time()))
                rc, health_text = capture_command(
                    [xpu_smi, "health", "-l"], health_path
                )
                critical = health_is_critical(health_text)
                write_event("health_poll", exit_code=rc, critical=critical,
                            path=str(health_path))
                if critical:
                    health_critical_observed = True
                    stop_admissions = True
                    print("Critical XPU health detected; stopping new admissions", flush=True)
                next_health_time = time.monotonic() + health_interval

            finished_tiles = []
            for tile, run in list(active.items()):
                exit_code = run.process.poll()
                if exit_code is None:
                    continue
                run.log_handle.close()
                ended_utc = utc_now()
                duration = time.monotonic() - run.started_wall
                failure_class = "" if exit_code == 0 else classify_failure(exit_code, run.log_path)
                retry = False
                status = "succeeded"
                if exit_code != 0:
                    status = "failed"
                    failure_counts[failure_class] += 1
                    if failure_class == "level_zero":
                        run.slot.quarantined = True
                        quarantined_tiles.add(run.slot.tile)
                        if run.task.attempt <= retries and any(
                                not slot.quarantined and slot.tile != run.slot.tile
                                for slot in slots):
                            tasks.append(Task(
                                run.task.task_id,
                                run.task.dataset,
                                run.task.horizon,
                                run.task.seed,
                                run.task.attempt + 1,
                            ))
                            retry = True
                            status = "retrying"
                    elif failure_class.startswith("signal_"):
                        # An unhandled signal may have bypassed interpreter and
                        # Level Zero context teardown. Do not add more contexts
                        # to this node; drain already-active experiments only.
                        run.slot.quarantined = True
                        quarantined_tiles.add(run.slot.tile)
                        stop_admissions = True
                        write_event(
                            "abrupt_child_exit", task_id=run.task.task_id,
                            tile=run.slot.tile, failure_class=failure_class,
                            action="stop_admissions_and_drain",
                        )

                record = {
                    "run_id": run_id,
                    "task_id": run.task.task_id,
                    "model": args.model,
                    "action": args.action,
                    "dataset": run.task.dataset,
                    "horizon": run.task.horizon,
                    "seed": run.task.seed,
                    "attempt": run.task.attempt,
                    "tile": tile,
                    "cpu_set": run.slot.cpu_set,
                    "pid": run.process.pid,
                    "started_utc": run.started_utc,
                    "ended_utc": ended_utc,
                    "duration_seconds": "{:.3f}".format(duration),
                    "exit_code": exit_code,
                    "status": status,
                    "failure_class": failure_class,
                    "retry_scheduled": int(retry),
                    "attempt_log": str(run.log_path),
                }
                write_attempt(attempts_writer, attempts_handle, record)
                write_event("attempt_finished", **record)
                if not retry:
                    final_status[run.task.task_id] = status
                print(
                    "[{}] {} {} on tile {} in {:.1f}s{}".format(
                        status.upper(), run.task.task_id, "attempt {}".format(run.task.attempt),
                        tile, duration,
                        " ({})".format(failure_class) if failure_class else "",
                    ),
                    flush=True,
                )
                finished_tiles.append(tile)

            for tile in finished_tiles:
                del active[tile]

            healthy_free = [
                slot for slot in slots
                if not slot.quarantined and slot.tile not in active
            ]
            if tasks and not active and not healthy_free:
                stop_admissions = True

            if stop_admissions and not active:
                while tasks:
                    task = tasks.popleft()
                    final_status.setdefault(task.task_id, "not_started")
                break

            if tasks and healthy_free and not stop_admissions and now >= next_launch_time:
                task = tasks.popleft()
                slot = healthy_free[0]
                log_path = attempts_dir / "{}.attempt{}.tile{}.log".format(
                    task.task_id, task.attempt, slot.tile
                )
                log_handle = log_path.open("w")
                child_env = os.environ.copy()
                child_env.pop("ZE_AFFINITY_MASK", None)
                child_env["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
                child_env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"
                child_env["MODEL_ZE_AFFINITY_MASK"] = slot.tile
                child_env["MODEL_EXPECT_SINGLE_XPU"] = "1"
                child_env.setdefault("MODEL_NUM_WORKERS", "0")
                child_env.setdefault("MODEL_CPU_THREADS", "4")
                child_env.setdefault("MODEL_SKIP_COMPLETED", "1")

                if dry_run:
                    dry_run_seconds = os.environ.get("MODEL_DRY_RUN_SECONDS", "0.05")
                    if (os.environ.get("MODEL_DRY_RUN_ZE_FAILURE") == task.task_id
                            and task.attempt == 1):
                        command = [
                            "bash", "-c",
                            "printf 'ZE_RESULT_ERROR_DEVICE_LOST simulated for %s\\n' \"$1\"; exit 9",
                            "_", task.task_id,
                        ]
                    else:
                        command = [
                            "bash", "-c",
                            "printf 'DRY RUN tile=%s task=%s\\n' \"$MODEL_ZE_AFFINITY_MASK\" \"$1\"; sleep \"$2\"",
                            "_", task.task_id, dry_run_seconds,
                        ]
                elif args.task_kind == "pretrain":
                    command = [
                        "bash", "scripts/run_pretrain_worker.sh", args.model,
                        task.dataset, str(task.horizon), str(task.seed),
                    ]
                else:
                    command = [
                        "bash", "scripts/launch_model.sh", args.model, args.action,
                        task.dataset, str(task.horizon), str(task.seed),
                    ]
                if slot.cpu_set:
                    command = ["taskset", "-c", slot.cpu_set, *command]

                started_utc = utc_now()
                process = subprocess.Popen(
                    command,
                    cwd=repo_root,
                    env=child_env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active[slot.tile] = ActiveRun(
                    task=task,
                    slot=slot,
                    process=process,
                    log_handle=log_handle,
                    log_path=log_path,
                    started_wall=time.monotonic(),
                    started_utc=started_utc,
                )
                write_event(
                    "attempt_started", task_id=task.task_id, attempt=task.attempt,
                    tile=slot.tile, cpu_set=slot.cpu_set, pid=process.pid,
                    command=command, attempt_log=str(log_path), forced_signal="none",
                )
                print(
                    "[START] {} attempt {} -> tile {}{} (pid {})".format(
                        task.task_id, task.attempt, slot.tile,
                        " cpu {}".format(slot.cpu_set) if slot.cpu_set else "",
                        process.pid,
                    ),
                    flush=True,
                )
                next_launch_time = time.monotonic() + stagger

            time.sleep(0.1)
    finally:
        # An unexpected scheduler exception must not orphan initialized XPU
        # children. Wait without sending TERM/KILL; this can intentionally
        # block if a child is hung because forced teardown is not ze_peak-safe.
        for tile, run in list(active.items()):
            if run.process.poll() is None:
                print(
                    "Scheduler cleanup is waiting for tile {} pid {}; "
                    "no force signal will be sent".format(tile, run.process.pid),
                    flush=True,
                )
                run.process.wait()
            if not run.log_handle.closed:
                run.log_handle.close()
            write_event(
                "child_reaped_in_scheduler_cleanup", task_id=run.task.task_id,
                tile=tile, pid=run.process.pid,
                exit_code=run.process.returncode, forced_signal="none",
            )
        active.clear()
        attempts_handle.close()
        if monitor_enabled and xpu_smi:
            sample_telemetry(xpu_smi, telemetry_path)
        summarize_telemetry(
            run_dir / "xpu_smi_telemetry.csv",
            run_dir / "xpu_smi_summary.json",
        )
        if monitor_enabled and xpu_smi:
            rc, health_text = capture_command(
                [xpu_smi, "health", "-l"], run_dir / "xpu_health_end.txt"
            )
            end_critical = health_is_critical(health_text)
            health_critical_observed = health_critical_observed or end_critical
            write_event("health_end", exit_code=rc, critical=end_critical)

    counts = Counter(final_status.values())
    overall = "succeeded"
    if received_signal is not None:
        overall = "interrupted_gracefully"
    elif counts.get("failed") or counts.get("not_started"):
        overall = "failed"
    elif health_critical_observed:
        overall = "completed_with_critical_health"
    elif quarantined_tiles:
        overall = "completed_with_device_quarantine"
    safety_status = "clean_lifecycle"
    if any(name.startswith("signal_") for name in failure_counts):
        safety_status = "unsafe_abrupt_child_exit"
    elif health_critical_observed:
        safety_status = "critical_health"
    elif quarantined_tiles:
        safety_status = "device_attention_required"
    elif received_signal is not None:
        safety_status = "graceful_interruption"
    summary = {
        "run_id": run_id,
        "status": overall,
        "ended_utc": utc_now(),
        "received_signal": received_signal,
        "forced_signals_sent": 0,
        "safety_status": safety_status,
        "failure_counts": dict(failure_counts),
        "critical_health_observed": health_critical_observed,
        "task_status_counts": dict(counts),
        "task_status": final_status,
        "quarantined_tiles": sorted(quarantined_tiles, key=int),
        "attempts_csv": str(run_dir / "attempts.csv"),
        "telemetry": str(run_dir / "xpu_smi_telemetry.csv"),
    }
    atomic_json(run_dir / "summary.json", summary)
    print(
        "Benchmark {}: {}; tasks {}; quarantined tiles {}; summary {}".format(
            run_id, overall, dict(counts), summary["quarantined_tiles"],
            run_dir / "summary.json",
        ),
        flush=True,
    )
    return 0 if overall == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
