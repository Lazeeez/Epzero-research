import json
import shutil

import numpy as np
import pytest

from research.longeval import vamana_screen

pytestmark = pytest.mark.skipif(shutil.which('g++') is None, reason='needs g++')


def _sample(root, n=3000, d=24, queries=60, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    q = rng.standard_normal((queries, d)).astype(np.float32)
    dist = ((q[:, None, :].astype(np.float64) - v[None]) ** 2).sum(-1)
    root.mkdir()
    np.save(root / 'vectors.npy', v)
    np.save(root / 'queries.npy', q)
    np.save(root / 'exact-top100.npy', np.argsort(dist, axis=1, kind='stable')[:, :100])
    np.save(root / 'cohort.npy', np.repeat([1, 2, 3], n // 3).astype(np.int8))
    np.save(root / 'lexical-rank.npy', np.arange(n))
    (root / 'sample.json').write_text(json.dumps({'metric': 'l2'}))
    return v, q, dist


def test_driver_finds_true_neighbours_and_bounds_degree(tmp_path):
    _, _, dist = _sample(tmp_path / 'sample')
    report = vamana_screen.run(tmp_path / 'sample', tmp_path / 'build', order='shuffled', seed=11, cache=tmp_path / 'cache')
    assert report['over_degree'] == 0
    found = np.fromfile(tmp_path / 'build' / 'labels-L200.bin', dtype='<i4').reshape(60, 100)
    truth = np.argsort(dist, axis=1, kind='stable')[:, :10]
    recall10 = np.mean([len(set(found[i, :10]) & set(truth[i])) / 10 for i in range(60)])
    assert recall10 > 0.99
    outdegree = np.fromfile(tmp_path / 'build' / 'outdegree.bin', dtype='<i4')
    assert outdegree.max() <= vamana_screen.R
    table = np.load(tmp_path / 'build' / 'per-query-ef200.npy')
    assert table.shape == (60, 3, 3, 2) and (table[..., 0].sum(axis=(1, 2)) == 100).all()


def test_driver_rejects_unnormalised_inner_product(tmp_path):
    _sample(tmp_path / 'sample')
    (tmp_path / 'sample' / 'sample.json').write_text(json.dumps({'metric': 'ip'}))
    with pytest.raises(ValueError):
        vamana_screen.run(tmp_path / 'sample', tmp_path / 'build', order='appended', seed=11, cache=tmp_path / 'cache')
