"""Create a deterministic, small forecasting dataset for integration tests."""

import argparse
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--rows', type=int, default=720)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2020, 1, 1)
    with args.output.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['date', 'signal_a', 'signal_b', 'OT'])
        for index in range(args.rows):
            writer.writerow([
                (start + timedelta(hours=index)).isoformat(sep=' '),
                math.sin(index / 12.0),
                math.cos(index / 17.0),
                0.4 * math.sin(index / 12.0)
                + 0.6 * math.cos(index / 31.0),
            ])


if __name__ == '__main__':
    main()
