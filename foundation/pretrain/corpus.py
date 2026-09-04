"""Curated Tier-1 reads. No corpus discovery, re-splitting, or value copying.

Weight/cap formulas follow AgenticTimer's DSM adapter and the integration note.
Index caches are content-addressed; draws live in a checkpointed parent RNG.
"""
from __future__ import annotations

import hashlib
import json
import os
import fcntl
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATA = Path('/lus/flare/projects/FRAME-IDP/siebenschuh/TimeSeriesData/msm')
REGISTRY = {
    'v1': (DATA / 'tier1/MANIFEST.json', None),
    'v2': (DATA / 'enhanced_pretrain_v2/wave1/MANIFEST_10B.json',
           DATA / 'enhanced_pretrain_v2/holdout_25M'),
    'v3': (DATA / 'extensive_pretrain_v3/MANIFEST_V3.json',
           DATA / 'extensive_pretrain_v3/holdout_25M'),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stable_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'little')


def weights(entries, shares):
    totals, stored = Counter(), Counter()
    for e in entries:
        f = e.get('family', '')
        totals[f] += e['n_series']
        stored[f] += e.get('weight', 0.)
    return {e['rel_path']: shares.get(e.get('family', ''), stored[e.get('family', '')])
            * e['n_series'] / totals[e.get('family', '')] for e in entries}


class Source:
    def __init__(self, root, rel, entry=None):
        self.rel = rel
        self.path = Path(root) / rel
        for name in ('metadata.json', 'splits.json', 'series_index.parquet', 'y_values.npy'):
            if not (self.path / name).is_file():
                raise FileNotFoundError(self.path / name)
        self.meta = json.loads((self.path / 'metadata.json').read_text())
        raw = json.loads((self.path / 'splits.json').read_text())
        self.counts = {s: int(raw.get('n_' + s, raw.get(s, 0)))
                       for s in ('train', 'val', 'test', 'excluded')}
        if entry and (int(entry['length']) != self.meta['length'] or
                      int(entry['C_y']) != self.meta['C_y']):
            raise ValueError(f'{rel}: metadata disagrees with manifest geometry')
        if self.meta['C_y'] != 1 or self.meta.get('total_T_irreg', 0):
            raise ValueError(f'{rel}: this pretraining contract requires regular univariate rows')
        self.values = np.load(self.path / 'y_values.npy', mmap_mode='r', allow_pickle=False)
        if self.values.ndim != 2 or self.values.shape[1] != 1 or self.values.dtype != np.float32:
            raise ValueError(f'{rel}: expected float32 (total_T, 1) storage')
        self.fingerprint = {
            'index_sha256': sha256(self.path / 'series_index.parquet'),
            'metadata_sha256': sha256(self.path / 'metadata.json'),
            'splits_sha256': sha256(self.path / 'splits.json'),
            'values_size': (self.path / 'y_values.npy').stat().st_size,
            'values_mtime_ns': (self.path / 'y_values.npy').stat().st_mtime_ns,
        }

    def index(self, split, cap, seed, cache_root):
        """Validate all rows, then use prefix of a fixed source permutation.

        Only the selected offset/length pairs are retained after preprocessing.
        A shared disk cache avoids rebuilding them for every horizon and seed.
        """
        identity = json.dumps([str(self.path), self.fingerprint, split, cap, seed,
                               'offset-index-v1'], sort_keys=True)
        key = hashlib.sha256(identity.encode()).hexdigest()
        cache_root = Path(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        target = cache_root / (key + '.npy')
        with (cache_root / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                print(f'Preparing {self.rel}: {split}, cap={cap}', flush=True)
                count = self.counts[split]
                rows = np.empty((count, 2), dtype=np.int64)
                cursor = 0
                actual = Counter()
                file = pq.ParquetFile(self.path / 'series_index.parquet')
                required = {'offset', 'length', 'split', 'group_id', 'regular', 'C_y'}
                if not required.issubset(file.schema_arrow.names):
                    raise ValueError(f'{self.rel}: missing index columns {required}')
                for batch in file.iter_batches(columns=sorted(required), batch_size=131072):
                    d = batch.to_pydict()
                    off, length = np.asarray(d['offset']), np.asarray(d['length'])
                    if (np.any(off < 0) or np.any(length != self.meta['length']) or
                            np.any(off + length > len(self.values)) or
                            any(x != 1 for x in d['C_y']) or not all(d['regular'])):
                        raise ValueError(f'{self.rel}: invalid/irregular row geometry')
                    actual.update(d['split'])
                    take = np.asarray(d['split']) == split
                    n = int(take.sum())
                    if cursor + n > count:
                        raise ValueError(f'{self.rel}: split counts changed')
                    rows[cursor:cursor+n, 0] = off[take]
                    rows[cursor:cursor+n, 1] = length[take]
                    cursor += n
                if dict(actual) != {k: v for k, v in self.counts.items() if v}:
                    raise ValueError(f'{self.rel}: index/splits disagreement {actual} vs {self.counts}')
                order = np.random.default_rng(stable_seed(seed, self.rel)).permutation(count)
                selected = rows[order[:cap]]
                tmp = target.with_suffix(f'.{os.getpid()}.tmp')
                with tmp.open('wb') as f:
                    np.save(f, selected, allow_pickle=False)
                os.replace(tmp, target)
        return np.load(target, mmap_mode='r', allow_pickle=False), key


class Corpus:
    def __init__(self, version, context, horizon, budget_obs, subset_seed, cache_root,
                 manifest=None, source_root=None, holdout_root=None, val_windows=1024):
        default_manifest, default_holdout = REGISTRY[version]
        manifest = Path(manifest or default_manifest)
        m = json.loads(manifest.read_text())
        root = Path(source_root or m['tier1_root'])
        entries = m['sources']
        if len(entries) != int(m.get('n_sources', len(entries))):
            raise ValueError('manifest source count mismatch')
        if len({e['rel_path'] for e in entries}) != len(entries):
            raise ValueError('duplicate manifest source identities')
        selected = entries
        exclusions = []
        if version == 'v1':
            wanted = {'gifteval_uni', 'kernelsynth'}
            selected = [e for e in entries if e['rel_path'] in wanted]
            if {e['rel_path'] for e in selected} != wanted:
                raise ValueError('v1 requires explicit main gifteval_uni and kernelsynth sources')
            exclusions = [{'source': e['rel_path'], 'reason': 'outside v1 main regular univariate scope'}
                          for e in entries if e not in selected]
        self.context, self.horizon = context, horizon
        self.sources = [Source(root, e['rel_path'], e) for e in selected]
        if any(s.meta['length'] != 1024 for s in self.sources):
            raise ValueError('selected corpus sources must have 1024-point rows')
        if context < 2 or horizon < 1 or context + horizon > 1024:
            raise ValueError('require context >= 2, horizon >= 1, context+horizon <= 1024')
        train_total = sum(s.counts['train'] for s in self.sources)
        if not train_total:
            raise ValueError('no training rows')
        budget_windows = int(budget_obs) // 1024
        if budget_windows < 1:
            raise ValueError('budget must include at least 1024 observation slots')
        base_weights = weights(entries, m.get('family_shares', {})) if version != 'v1' else {}
        self.indices, kept, sampling = [], [], []
        records = []
        for s in self.sources:
            cap = min(s.counts['train'], s.counts['train'] * budget_windows // train_total)
            if not cap:
                exclusions.append({'source': s.rel, 'reason': 'zero train rows or zero budget cap'})
                continue
            idx, subset_id = s.index('train', cap, subset_seed, cache_root)
            self.indices.append(idx)
            kept.append(s)
            sampling.append(base_weights[s.rel] if base_weights else s.counts['train'])
            records.append({'source': s.rel, 'train_rows': s.counts['train'], 'cap': cap,
                            'subset_id': subset_id, **s.fingerprint})
        self.sources = kept
        self.probabilities = np.asarray(sampling, dtype=np.float64)
        if not len(kept) or self.probabilities.sum() <= 0:
            raise ValueError('budget/mixture leaves no eligible sources')
        self.probabilities /= self.probabilities.sum()
        for r, w in zip(records, self.probabilities):
            r['sampling_weight'] = float(w)
        # v1 preserves per-source val; v2/v3 use their own standalone val export.
        val_sources = self.sources if version == 'v1' else [Source(holdout_root or default_holdout, 'val')]
        self.validation = []
        for s in val_sources:
            if s.counts['val']:
                idx, key = s.index('val', min(val_windows, s.counts['val']), subset_seed, cache_root)
                self.validation.append((s, idx, key))
        if not self.validation:
            raise ValueError('no curated validation rows; no random carve is permitted')
        self.audit = {'version': version, 'corpus_id': m.get('corpus_id'),
                      'manifest': str(manifest), 'manifest_sha256': sha256(manifest),
                      'source_root': str(root), 'selected_sources': records, 'exclusions': exclusions,
                      'budget_obs_requested': int(budget_obs), 'budget_windows_requested': budget_windows,
                      'realized_windows': sum(len(i) for i in self.indices),
                      'realized_slots': 1024 * sum(len(i) for i in self.indices),
                      'validation': [{'source': str(s.path), 'rows': len(i), 'subset_id': k,
                                      **s.fingerprint} for s, i, k in self.validation],
                      'budget_unit': 'stored scalar slots (v3 includes padding)',
                      'sampling': 'weighted source then uniform row, with replacement'}

    def crop(self, source, index, row, rng=None):
        off, length = map(int, index[row])
        width = self.context + self.horizon
        start = length - width if rng is None else int(rng.integers(length - width + 1))
        # Always bound by stored offset AND length, never a global row stride.
        return np.array(source.values[off+start:off+start+width], copy=True)

    def batch(self, size, rng):
        choices = rng.choice(len(self.sources), size=size, p=self.probabilities)
        windows = [self.crop(self.sources[i], self.indices[i],
                             int(rng.integers(len(self.indices[i]))), rng) for i in choices]
        return np.stack(windows), choices


def normalize(windows, context):
    """Context-only finite mean/std, zero imputation, masked targets. No clipping.

    Float64 arithmetic protects statistics on large finite magnitudes. Invalid
    contexts/targets are explicitly skipped and counted by the trainer.
    """
    a = windows.astype(np.float64)
    mask = np.isfinite(a)
    safe = np.where(mask, a, 0.)
    count = mask[:, :context].sum(axis=1, keepdims=True)
    mean = safe[:, :context].sum(axis=1, keepdims=True) / np.maximum(count, 1)
    diff = np.where(mask[:, :context], a[:, :context] - mean, 0.)
    var = (diff * diff).sum(axis=1, keepdims=True) / np.maximum(count - 1, 1)
    scale = np.sqrt(np.maximum(var, 1e-10))
    z = np.where(mask, (safe - mean) / scale, 0.)
    valid = (count[:, 0, 0] >= 2) & (mask[:, context:].sum(axis=(1, 2)) > 0)
    valid &= (np.abs(z).max(axis=(1, 2)) < 1e15)
    z[~valid] = 0.
    return z.astype(np.float32), mask, valid
