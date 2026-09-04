"""Single-device corpus trainer with atomic full-state restart checkpoints."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from foundation.pretrain.corpus import Corpus, normalize, sha256
from foundation.pretrain.models import build_model, forecast, IMPLEMENTED
from utils.graceful_shutdown import install_signal_handlers, requested_signal

CHECKPOINT_ROOT = '/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesTraining/FITS_checkpoints/pretrained'


def env_number(name, default, kind=int):
    return kind(float(os.environ.get(name, default)))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=IMPLEMENTED, required=True)
    p.add_argument('--corpus', choices=('v1', 'v2', 'v3'), required=True)
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--context', type=int, default=env_number('PRETRAIN_CONTEXT', 512))
    p.add_argument('--budget-obs', type=lambda s: int(float(s)), default=env_number('PRETRAIN_BUDGET_OBS', 1e9))
    p.add_argument('--steps', type=int, default=env_number('PRETRAIN_STEPS', 10000))
    p.add_argument('--batch-size', type=int, default=env_number('PRETRAIN_BATCH_SIZE', 256))
    p.add_argument('--cutoff', type=int, default=env_number('PRETRAIN_CUT_FREQ', 64))
    p.add_argument('--lr', type=float, default=env_number('PRETRAIN_LR', .0005, float))
    p.add_argument('--subset-seed', type=int, default=env_number('PRETRAIN_SUBSET_SEED', 1234))
    p.add_argument('--save-every', type=int, default=env_number('PRETRAIN_SAVE_EVERY', 100))
    p.add_argument('--validate-every', type=int, default=env_number('PRETRAIN_VALIDATE_EVERY', 500))
    p.add_argument('--val-windows', type=int, default=env_number('PRETRAIN_VAL_WINDOWS', 1024))
    p.add_argument('--log-every', type=int, default=env_number('PRETRAIN_LOG_EVERY', 10))
    p.add_argument('--device', choices=('cpu', 'xpu'), default=os.environ.get('PRETRAIN_DEVICE', 'xpu'))
    p.add_argument('--checkpoint-root', default=os.environ.get('PRETRAIN_CHECKPOINT_ROOT', CHECKPOINT_ROOT))
    p.add_argument('--cache-root', default=os.environ.get('PRETRAIN_INDEX_CACHE', CHECKPOINT_ROOT + '/_index_cache'))
    p.add_argument('--manifest', default=None)
    p.add_argument('--source-root', default=None)
    p.add_argument('--holdout-root', default=None)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    for field in ('steps', 'batch_size', 'save_every', 'validate_every', 'val_windows', 'log_every'):
        if getattr(args, field) < 1:
            p.error(f'{field} must be positive')
    if not args.lr > 0:
        p.error('learning rate must be positive')
    # Version-specific overrides let one invocation schedule v1/v2/v3 correctly.
    for field in ('manifest', 'source_root', 'holdout_root'):
        if getattr(args, field) is None:
            setattr(args, field, os.environ.get(f'PRETRAIN_{args.corpus.upper()}_{field.upper()}'))
    return args


def atomic_save(payload, path):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def tensor_batch(raw, context, device):
    z, mask, valid = normalize(raw, context)
    return (torch.from_numpy(z[valid]).to(device),
            torch.from_numpy(mask[valid]).to(device), valid)


def validation(model, corpus, size, device):
    model.eval()
    total, count, skipped = 0., 0, 0
    per_source = {}
    with torch.no_grad():
        for source, index, _ in corpus.validation:
            source_sum, source_count = 0., 0
            for start in range(0, len(index), size):
                raw = np.stack([corpus.crop(source, index, i)
                                for i in range(start, min(start+size, len(index)))])
                z, mask, valid = tensor_batch(raw, corpus.context, device)
                skipped += int((~valid).sum())
                if not len(z):
                    continue
                pred = forecast(model, z[:, :corpus.context], corpus.horizon)
                error = (pred - z[:, corpus.context:]).square()
                target_mask = mask[:, corpus.context:]
                source_sum += float(error[target_mask].sum().item())
                source_count += int(target_mask.sum().item())
            total += source_sum
            count += source_count
            per_source[source.rel] = {'squared_error': source_sum, 'finite_targets': source_count}
    model.train()
    if not count or not np.isfinite(total):
        raise RuntimeError('validation has no finite loss/targets')
    return {'mse': total/count, 'finite_targets': count, 'skipped_rows': skipped,
            'sources': per_source}


def main():
    args = arguments()
    install_signal_handlers()
    torch.set_num_threads(env_number('MODEL_CPU_THREADS', 4))
    # Validate architecture/geometry before expensive source preparation.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(args.model, args.context, args.horizon, args.cutoff)
    corpus = Corpus(args.corpus, args.context, args.horizon, args.budget_obs,
                    args.subset_seed, args.cache_root, args.manifest, args.source_root,
                    args.holdout_root, args.val_windows)
    config = {k: getattr(args, k) for k in ('model', 'corpus', 'horizon', 'seed', 'context',
              'budget_obs', 'steps', 'batch_size', 'cutoff', 'lr', 'subset_seed', 'val_windows',
              'validate_every')}
    config['objective'] = 'context-normalized masked forecast MSE; random within-row crops'
    config['code_sha256'] = {str(p.relative_to(Path.cwd())): sha256(p) for p in
                             (Path(__file__), Path(__file__).with_name('corpus.py'),
                              Path(__file__).with_name('models.py'), Path.cwd() / 'models/FITS/model.py')}
    identity = hashlib.sha256(json.dumps({'config': config, 'corpus': corpus.audit},
                                         sort_keys=True).encode()).hexdigest()
    setting = (f'pretrain_{args.corpus}_sl{args.context}_pl{args.horizon}_cf{args.cutoff}'
               f'_obs{args.budget_obs}_steps{args.steps}_s{args.seed}_{identity[:12]}')
    directory = Path(args.checkpoint_root) / args.model / setting
    directory.mkdir(parents=True, exist_ok=True)
    print(f'Pretraining checkpoint directory: {directory}', flush=True)
    print(f'Corpus: {corpus.audit["realized_windows"]} eligible windows; '
          f'{len(corpus.sources)} sources; sampling with replacement', flush=True)
    with (directory / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'pretraining already running: {directory}') from None
        (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (directory / 'corpus.json').write_text(json.dumps(corpus.audit, indent=2) + '\n')
        if args.prepare_only:
            return
        if (directory / 'complete.json').exists():
            print(f'Already completed; skipping {setting}', flush=True)
            return
        if requested_signal():
            raise SystemExit(128 + requested_signal())
        if args.device == 'xpu':
            if not torch.xpu.is_available():
                raise RuntimeError('XPU requested but unavailable; use a compute-node allocation')
            if os.environ.get('MODEL_EXPECT_SINGLE_XPU') == '1' and torch.xpu.device_count() != 1:
                raise RuntimeError('pool worker must see exactly one XPU')
        device = torch.device(args.device + ':0' if args.device == 'xpu' else 'cpu')
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        rng = np.random.default_rng(args.seed)
        step, best = 0, float('inf')
        stats = {s.rel: {'draws': 0, 'skipped_rows': 0, 'finite_slots': 0} for s in corpus.sources}
        last = directory / 'last.pth'

        def save():
            if args.device == 'xpu':
                torch.xpu.synchronize()
            atomic_save({'format_version': 1, 'identity': identity, 'config': config,
                         'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'step': step, 'best_val': best, 'draw_rng': rng.bit_generator.state,
                         'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                         'torch_rng': torch.get_rng_state(),
                         'xpu_rng': torch.xpu.get_rng_state() if args.device == 'xpu' else None,
                         'source_stats': stats, 'device_type': args.device}, last)

        try:
            if last.exists():
                # Trusted local checkpoints only: contains Python/NumPy RNG state.
                state = torch.load(last, map_location='cpu', weights_only=False)
                if state['identity'] != identity or state['device_type'] != args.device:
                    raise ValueError('checkpoint configuration/device identity mismatch')
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
                step, best, stats = state['step'], state['best_val'], state['source_stats']
                rng.bit_generator.state = state['draw_rng']
                random.setstate(state['python_rng'])
                np.random.set_state(state['numpy_rng'])
                torch.set_rng_state(state['torch_rng'])
                if args.device == 'xpu':
                    torch.xpu.set_rng_state(state['xpu_rng'])
                print(f'Resuming at completed step {step}', flush=True)
            model.train()
            started = time.monotonic()
            with (directory / 'training.jsonl').open('a', buffering=1) as log:
                while step < args.steps and not requested_signal():
                    raw, choices = corpus.batch(args.batch_size, rng)
                    z, mask, valid = tensor_batch(raw, args.context, device)
                    for i, s in enumerate(corpus.sources):
                        sel = choices == i
                        stats[s.rel]['draws'] += int(sel.sum())
                        stats[s.rel]['skipped_rows'] += int((sel & ~valid).sum())
                        stats[s.rel]['finite_slots'] += int(np.isfinite(raw[sel]).sum())
                    if not len(z):
                        raise RuntimeError('entire batch invalid; inspect corpus/source statistics')
                    optimizer.zero_grad(set_to_none=True)
                    pred = forecast(model, z[:, :args.context], args.horizon)
                    loss = (pred - z[:, args.context:]).square()[mask[:, args.context:]].mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite pretraining loss')
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise RuntimeError('nonfinite gradients')
                    optimizer.step()
                    step += 1
                    if step % args.log_every == 0 or step == 1:
                        record = {'step': step, 'loss': float(loss.item()), 'valid_rows': len(z),
                                  'elapsed_seconds': time.monotonic()-started}
                        print(json.dumps(record), flush=True)
                        log.write(json.dumps(record) + '\n')
                    if step % args.validate_every == 0 or step == args.steps:
                        result = validation(model, corpus, args.batch_size, device)
                        log.write(json.dumps({'step': step, 'validation': result}) + '\n')
                        if result['mse'] < best:
                            best = result['mse']
                            atomic_save(model.state_dict(), directory / 'checkpoint.pth')
                        print(f'Step {step}: curated validation MSE={result["mse"]:.8g}', flush=True)
                    if step % args.save_every == 0 or requested_signal() or step == args.steps:
                        save()
            save()
            (directory / 'source_stats.json').write_text(json.dumps(stats, indent=2) + '\n')
            if step == args.steps:
                (directory / 'complete.json').write_text(json.dumps({'step': step, 'best_val': best}) + '\n')
        finally:
            # Do not checkpoint half an optimizer update on arbitrary exceptions.
            # last.pth remains the last atomic, completed-step snapshot.
            if args.device == 'xpu':
                torch.xpu.synchronize()
            optimizer, model = None, None
            gc.collect()
            if args.device == 'xpu':
                torch.xpu.empty_cache()
            if requested_signal():
                print(f'GRACEFUL_XPU_SHUTDOWN cleanup complete: signal={requested_signal()}', flush=True)
        if requested_signal():
            raise SystemExit(128 + requested_signal())


if __name__ == '__main__':
    main()
