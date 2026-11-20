import importlib.util

import numpy as np
import pytest

from research.longeval.standard_graphs import StandardGraph

# hnswlib is an optional third implementation; its tests run only where it is installed.
KINDS = ['epzero', 'faiss_hnsw', pytest.param('hnswlib', marks=pytest.mark.skipif(
    importlib.util.find_spec('hnswlib') is None, reason='hnswlib is not installed'))]


@pytest.mark.parametrize('kind', KINDS)
def test_ip_scores_labels_and_repeatability(kind):
    data = np.eye(8, dtype=np.float32)
    labels = np.array([91, 7, 23, 1004, 12, 81, 75, 20], dtype=np.int64)
    queries = np.array([[.7, -.3, .1, .9, -.2, .6, .4, .5],
                        [-.2, .1, .3, -.7, .6, .4, .9, .8]], dtype=np.float32)
    backend = StandardGraph(kind, data, labels, M=4, ef_construction=32)
    rows = backend.measure(queries, k=4, ef=16)
    for query, row in zip(queries, rows):
        expected = np.argsort(-(data @ query))[:4]
        assert [v for v, _ in row['results']] == list(labels[expected])
        np.testing.assert_allclose([v for _, v in row['results']], (data @ query)[expected], atol=1e-7)
        assert len(row['native_call_nanoseconds']) == 3
        assert min(row['native_call_nanoseconds']) > 0
    assert backend.measure(queries, k=4, ef=16)[0]['results'] == rows[0]['results']
    with pytest.raises(ValueError, match='k or ef'):
        backend.measure(queries, k=9, ef=16)
    with pytest.raises(ValueError, match='query'):
        backend.measure(np.ones((1, 9), dtype=np.float32), k=4, ef=16)


@pytest.mark.parametrize('kind', KINDS)
def test_rejects_duplicate_ids_nonfinite_and_padding(kind):
    data = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match='inputs'):
        StandardGraph(kind, data, np.array([7, 7, 9], dtype=np.int64))
    bad = data.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='inputs'):
        StandardGraph(kind, bad, np.arange(3, dtype=np.int64))
    backend = StandardGraph(kind, data, np.arange(3, dtype=np.int64), M=2, ef_construction=8)
    with pytest.raises(ValueError):
        backend.decode((np.array([[-1]]), np.array([[-1]])), 1)
