"""Small CPU acceptance tests; no environment changes or corpus writes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from foundation.pretrain.corpus import Source, Corpus, normalize, weights


def source(root, name, length=1024):
    p = root / name
    p.mkdir(parents=True)
    n = 24
    rng = np.random.default_rng(17)
    y = rng.normal(size=(n*length, 1)).astype(np.float32)
    y[:8] = np.nan
    np.save(p/'y_values.npy', y)
    split = ['train']*16 + ['val']*4 + ['test']*4
    pq.write_table(pa.table({'offset': np.arange(n)*length, 'length': [length]*n,
                            'split': split, 'group_id': np.arange(n),
                            'regular': [True]*n, 'C_y': [1]*n}), p/'series_index.parquet')
    (p/'metadata.json').write_text(json.dumps({'length': length, 'C_y': 1}))
    (p/'splits.json').write_text(json.dumps({'n_train': 16, 'n_val': 4, 'n_test': 4}))
    return {'rel_path': name, 'length': length, 'C_y': 1, 'n_series': n}


class Integration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fits_pretrain_test_')
        self.root = Path(self.temp.name)
        entries = [source(self.root, x) for x in ('gifteval_uni', 'kernelsynth')]
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text(json.dumps({'sources': entries, 'tier1_root': str(self.root)}))

    def tearDown(self):
        self.temp.cleanup()

    def corpus(self, budget):
        return Corpus('v1', 8, 4, budget, 1234, self.root/'cache', manifest=self.manifest,
                      val_windows=4)

    def test_boundaries_budgets_and_splits(self):
        for length in (512, 1024, 2048):
            entry = source(self.root, f'geometry{length}', length)
            s = Source(self.root, entry['rel_path'], entry)
            index, _ = s.index('train', 16, 10, self.root/'cache')
            self.assertTrue(np.all(index[:, 1] == length))
            self.assertTrue(np.all(index[:, 0] < 16*length))
        small, large = self.corpus(8*1024), self.corpus(16*1024)
        for a, b in zip(small.indices, large.indices):
            np.testing.assert_array_equal(a, b[:len(a)])
        rng1, rng2 = np.random.default_rng(7), np.random.default_rng(7)
        a, _ = small.batch(32, rng1)
        b, _ = small.batch(32, rng2)
        np.testing.assert_array_equal(a, b)
        c, _ = small.batch(32, rng2)
        self.assertFalse(np.array_equal(b, c))

    def test_masks_and_weights(self):
        x = np.array([[[np.nan], [2.], [4.], [np.inf], [8.], [np.nan]]])
        z, mask, valid = normalize(x, 4)
        self.assertTrue(valid[0])
        self.assertTrue(np.isfinite(z).all())
        self.assertEqual(z[0, 0, 0], 0.)
        self.assertFalse(mask[0, 3, 0])
        a = [{'rel_path':'large', 'family':'f', 'n_series':999, 'weight':1.},
             {'rel_path':'tiny', 'family':'f', 'n_series':1, 'weight':0.}]
        self.assertAlmostEqual(weights(a, {})['tiny'], .001)

    def test_versioned_holdouts_and_excluded_rows(self):
        entry = source(self.root, 'versioned')
        entry.update(family='real', weight=1.)
        p = self.root/'versioned'
        table = pq.read_table(p/'series_index.parquet')
        labels = ['train']*15 + ['excluded'] + ['val']*4 + ['test']*4
        table = table.set_column(table.schema.get_field_index('split'), 'split', pa.array(labels))
        pq.write_table(table, p/'series_index.parquet')
        (p/'splits.json').write_text(json.dumps(
            {'n_train': 15, 'n_excluded': 1, 'n_val': 4, 'n_test': 4}))
        holdout = self.root/'holdout'
        source(holdout, 'val')
        manifest = self.root/'versioned.json'
        manifest.write_text(json.dumps({'sources': [entry]}))
        for version in ('v2', 'v3'):
            corpus = Corpus(version, 8, 4, 1024*24, 1234, self.root/'cache',
                            manifest=manifest, source_root=self.root,
                            holdout_root=holdout, val_windows=4)
            self.assertEqual(len(corpus.indices[0]), 15)
            self.assertNotIn(15*1024, corpus.indices[0][:, 0])
            self.assertEqual(corpus.validation[0][0].path, holdout/'val')

    def command(self, output):
        return [sys.executable, '-u', '-m', 'foundation.pretrain.train',
                '--model', 'FITS', '--corpus', 'v1', '--manifest', str(self.manifest),
                '--context', '8', '--horizon', '4', '--cutoff', '3', '--seed', '114',
                '--budget-obs', '16384', '--steps', '20', '--batch-size', '4',
                '--save-every', '1', '--validate-every', '10', '--val-windows', '4',
                '--log-every', '1', '--device', 'cpu', '--cache-root', str(self.root/'cache'),
                '--checkpoint-root', str(output)]

    def test_interrupt_resume_matches_uninterrupted(self):
        env = dict(os.environ, MODEL_CPU_THREADS='1')
        full = self.root/'full'
        subprocess.run(self.command(full), env=env, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
        resumed = self.root/'resumed'
        proc = subprocess.Popen(self.command(resumed), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        lines = []
        for line in proc.stdout:
            lines.append(line)
            if '"step": 2,' in line:
                proc.send_signal(signal.SIGTERM)
                break
        rest, _ = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 143, ''.join(lines)+rest)
        subprocess.run(self.command(resumed), env=env, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
        a = torch.load(next(full.rglob('last.pth')), weights_only=False)
        b = torch.load(next(resumed.rglob('last.pth')), weights_only=False)
        self.assertEqual(a['step'], b['step'])
        self.assertEqual(a['draw_rng'], b['draw_rng'])
        for name in a['model']:
            torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
