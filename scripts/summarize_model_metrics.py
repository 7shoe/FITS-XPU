#!/usr/bin/env python3
"""Summarize standardized model metrics into CSV and Markdown comparison tables."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOT = "/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default=f"{DEFAULT_ROOT}/model_metrics.csv",
    help="merged metrics CSV created by merge_model_metrics.py",
)
parser.add_argument(
    "--output",
    default=f"{DEFAULT_ROOT}/model_summary.csv",
    help="aggregated comparison CSV",
)
parser.add_argument(
    "--markdown-output",
    default=f"{DEFAULT_ROOT}/model_summary.md",
    help="human-readable comparison table",
)
args = parser.parse_args()

with Path(args.input).open(newline="") as handle:
    source_rows = list(csv.DictReader(handle))

# Re-running evaluation appends another row.  Keep the most recently merged
# result for each exact seed/configuration rather than counting it twice.
latest = {}
for row in source_rows:
    key = tuple(
        row.get(field, "")
        for field in ("model", "data", "seq_len", "pred_len", "seed", "train_mode", "h_order")
    )
    latest[key] = row

groups = defaultdict(list)
for row in latest.values():
    key = tuple(row.get(field, "") for field in ("model", "data", "seq_len", "pred_len"))
    groups[key].append(row)

summary_rows = []
for (model, data, seq_len, pred_len), rows in sorted(groups.items()):
    def values(metric):
        return [float(row[metric]) for row in rows]

    mse = values("mse")
    mae = values("mae")
    summary_rows.append({
        "model": model,
        "data": data,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "runs": len(rows),
        "mse_mean": statistics.fmean(mse),
        "mse_std": statistics.pstdev(mse) if len(mse) > 1 else 0.0,
        "mae_mean": statistics.fmean(mae),
        "mae_std": statistics.pstdev(mae) if len(mae) > 1 else 0.0,
    })

fields = ["model", "data", "seq_len", "pred_len", "runs", "mse_mean", "mse_std", "mae_mean", "mae_std"]
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary_rows)

markdown = Path(args.markdown_output)
markdown.parent.mkdir(parents=True, exist_ok=True)
with markdown.open("w") as handle:
    handle.write("| Model | Dataset | Look-back | Horizon | Runs | MSE | MAE |\n")
    handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
    for row in summary_rows:
        handle.write(
            "| {model} | {data} | {seq_len} | {pred_len} | {runs} | "
            "{mse_mean:.6f} ± {mse_std:.6f} | "
            "{mae_mean:.6f} ± {mae_std:.6f} |\n".format(**row)
        )

print("Wrote {} summary rows to {} and {}".format(len(summary_rows), output, markdown))
