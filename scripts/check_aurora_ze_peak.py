#!/usr/bin/env python3
"""Record post-run Aurora node state for a benchmark-pool host.

This is a recurrence tripwire, not a health proof: a ze_peak prologue failure
only becomes visible after another job lands on the host, and node comments are
mutable.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path


DEFAULT_LEDGER = Path(
    "/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/"
    "ze_peak_ledger.jsonl"
)


def pbsnodes_snapshot():
    completed = subprocess.run(
        ["pbsnodes", "-avS"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "pbsnodes failed with {}: {}".format(
                completed.returncode, completed.stderr.strip()
            )
        )
    states = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].startswith("x"):
            states[parts[0].split(".")[0]] = {
                "state": parts[1] if len(parts) > 1 else "unknown",
                "line": line.strip(),
            }
    return states


def append_ledger(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()

    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    host = str(manifest["host"]).split(".")[0]
    state = pbsnodes_snapshot().get(
        host, {"state": "unknown", "line": "host absent from pbsnodes output"}
    )
    lowered = state["line"].lower()
    bad_state = any(
        token in state["state"].lower()
        for token in ("down", "offline", "state-unknown", "unknown")
    )
    ze_peak = "zepeak" in lowered or "ze_peak" in lowered
    record = {
        "time": time.time(),
        "run_id": manifest["run_id"],
        "pbs_jobid": manifest.get("pbs_jobid"),
        "host": host,
        "state": state["state"],
        "pbsnodes_line": state["line"],
        "bad_state": bad_state,
        "ze_peak": ze_peak,
        "caveat": "green means nothing visible yet, not proof of clean teardown",
    }
    append_ledger(args.ledger, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    if ze_peak:
        print("ZE_PEAK RECURRENCE: preserve this ledger entry and notify ALCF")
        return 1
    if bad_state:
        print("Host is unhealthy, but no ze_peak provenance is visible")
        return 1
    print("Nothing visible yet; repeat after the host receives a later job")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
