import json

import numpy as np
import pytest

from research.longeval import cross_pilot, insertion_pilot, insertion_strata
from research.longeval.tests.test_post_freeze_analyses import small_sample


def _with_confirmation(sample):
    vectors = np.load(sample / 'vectors.npy')
    confirm = np.random.default_rng(9).standard_normal((5, vectors.shape[1])).astype(np.float32)
    np.save(sample / 'confirm-queries.npy', confirm)
    np.save(sample / 'confirm-exact-top100.npy', np.argsort(-(confirm.astype(np.float64) @ vectors.astype(np.float64).T),
                                                           axis=1, kind='stable')[:, :100].astype(np.int32))
    return sample


@pytest.mark.parametrize('order', ['appended', 'reversed', 'shuffled'])
def test_it_reproduces_the_frozen_pilot_exactly_on_faiss(tmp_path, order):
    pytest.importorskip('faiss')
    sample = _with_confirmation(small_sample(tmp_path / 'sample'))
    settings = dict(order=order, seed=11, ef=40, M=8, ef_construction=40, efs=(10, 40, 80))
    frozen = insertion_pilot.run(sample, tmp_path / 'frozen', **settings)
    mine = cross_pilot.run(sample, tmp_path / 'mine', backend='faiss_hnsw', **settings)
    files = sorted(p.name for p in (tmp_path / 'frozen').iterdir())
    assert files == sorted(p.name for p in (tmp_path / 'mine').iterdir())
    for name in files:
        if name.endswith('.npy'):
            assert np.array_equal(np.load(tmp_path / 'frozen' / name), np.load(tmp_path / 'mine' / name)), name
    frozen.pop('build_seconds'), mine.pop('build_seconds')
    assert json.loads(json.dumps(frozen)) == json.loads(json.dumps(mine))


def test_hnswlib_builds_score_and_feed_the_strata_analysis(tmp_path):
    pytest.importorskip('hnswlib')
    sample = small_sample(tmp_path / 'sample')
    for order in ('appended', 'reversed', 'shuffled'):
        report = cross_pilot.run(sample, tmp_path / 'hnswlib' / f'{order}-11', order=order, seed=11, ef=40, M=8,
                                 ef_construction=40, efs=(40, 200))
        assert report['backend'] == 'hnswlib' and report['cohorts']['1']['indegree_mean'] is None
        assert report['widths']['200']['recall_at_100'] > 0.9
        labels = np.load(tmp_path / 'hnswlib' / f'{order}-11' / 'labels-ef200.npy')
        assert labels.shape == (12, 100) and labels.min() >= 0
    out = insertion_strata.orders(tmp_path / 'hnswlib', ef=200, draws=50)
    assert set(out['cohort3_vs_cohort1']['within_query_ratio']) >= {'appended', 'reversed', 'shuffled'}


def test_an_unknown_backend_is_refused(tmp_path):
    with pytest.raises(ValueError):
        cross_pilot.run(small_sample(tmp_path / 'sample'), tmp_path / 'x', order='appended', seed=11, backend='lucene')
