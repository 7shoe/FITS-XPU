#!/usr/bin/env python3
"""Merge standardized per-model metrics CSVs into one comparison table."""

import argparse
import csv
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument(
    "--search-root",
    default="/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining",
    help="directory containing per-model *_results directories",
)
parser.add_argument(
    "--output",
    default="/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/model_metrics.csv",
    help="merged CSV path",
)
args = parser.parse_args()

output = Path(args.output).resolve()
metric_files = sorted(
    path for path in Path(args.search_root).glob("*_results/*/**/metrics.csv")
    if path.resolve() != output
)
rows = []
fields = []
seen_rows = set()
for metric_file in metric_files:
    with metric_file.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row_key = tuple(sorted(row.items()))
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            rows.append(row)
            for field in row:
                if field not in fields:
                    fields.append(field)

if not rows:
    raise SystemExit("No metrics.csv files found below {}".format(args.search_root))

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print("Merged {} rows from {} files into {}".format(len(rows), len(metric_files), output))
