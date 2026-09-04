"""Validate a pretraining matrix and run each implemented model through the pool."""
import os
from pathlib import Path
import signal
import subprocess
import sys

from foundation.pretrain.models import IMPLEMENTED


def main():
    args = sys.argv[1:]
    check = bool(args and args[0] == '--check')
    if check:
        args.pop(0)
    # Accept the initial proposed `MODEL data v1` spelling as an alias.
    delimiter = 'train' if 'train' in args else 'data'
    if delimiter not in args:
        raise SystemExit('Expected MODEL [MODEL ...] train v1 [v2 v3]')
    pos = args.index(delimiter)
    models, versions = args[:pos], args[pos+1:]
    if not models or not versions or any(v not in ('v1', 'v2', 'v3') for v in versions):
        raise SystemExit('Specify models, train, and one or more of v1 v2 v3')
    if len(set(models)) != len(models) or len(set(versions)) != len(versions):
        raise SystemExit('Duplicate models/corpora are not permitted')
    for model in models:
        if model not in IMPLEMENTED:
            raise SystemExit(f'{model}: pretraining adapter not implemented. Available: {IMPLEMENTED}')
    context = int(os.environ.get('PRETRAIN_CONTEXT', '512'))
    cutoff = int(os.environ.get('PRETRAIN_CUT_FREQ', '64'))
    horizons = [int(x) for x in os.environ.get('MODEL_HORIZONS', '96 192 336').split()]
    if not horizons or context < 2 or not 1 <= cutoff <= context // 2 + 1:
        raise SystemExit('Invalid context, cutoff, or empty horizon list')
    for horizon in horizons:
        output = context + horizon
        if horizon < 1 or output > 1024 or output % 2 or int(cutoff * output / context) > output // 2 + 1:
            raise SystemExit(f'Invalid FITS geometry: context={context}, horizon={horizon}, cutoff={cutoff}; '
                             'output must be even and fit within a 1024-slot corpus row')
    if check:
        return
    active = None
    interrupted = False

    def stop(signum, _frame):
        nonlocal interrupted
        interrupted = True
        if active is not None and active.poll() is None:
            active.send_signal(signum)  # pool drains children; never SIGKILL

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    root = Path(__file__).resolve().parents[2]
    for model in models:
        if interrupted:
            raise SystemExit(1)
        active = subprocess.Popen([
            sys.executable, '-u', str(root / 'scripts/run_benchmark_pool.py'),
            '--task-kind', 'pretrain', '--horizons', os.environ.get('MODEL_HORIZONS', '96 192 336'),
            '--seeds', os.environ.get('MODEL_SEEDS', '114 514 1919 810 0'),
            model, 'train', *versions], cwd=root)
        code = active.wait()
        if code or interrupted:
            raise SystemExit(code if code > 0 else 1)


if __name__ == '__main__':
    main()
