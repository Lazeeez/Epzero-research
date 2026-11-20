import json

import numpy as np
import pytest

from research.longeval.backends import Backend
from research.longeval.native_cost import CostObserver, native_cohort
from research.longeval.tests.test_trajectory import small_problem
from research.longeval.trajectory import trajectory_core


@pytest.mark.parametrize('kind', ['epzero', 'faiss_hnsw'])
def test_native_timing_reproduces_baselines_and_does_not_change_history(tmp_path, kind):
    specs, queries, refs, config = small_problem(tmp_path)
    baseline = trajectory_core(specs, queries, refs, tmp_path / 'baseline', config,
        kind=kind, mode='incremental', seed=11, provenance_record={'synthetic': True})
    observer = CostObserver(baseline, tmp_path / 'cost')
    timed = trajectory_core(specs, queries, refs, tmp_path / 'timed', config,
        kind=kind, mode='incremental', seed=11, provenance_record={'synthetic': True}, snapshot_observer=observer)
    for step, (a, b) in enumerate(zip(baseline['snapshots'], timed['snapshots']), 1):
        assert a['state'] == b['state']
        report = json.loads((tmp_path / f'cost/snapshot-{step}/report.json').read_text())
        assert report['status'] == 'completed'
        assert report['runs']['wide']['ann_recall_mean'] == 1.
        for result in report['runs'].values():
            assert len(result['queries']) == len(queries['development'].ids)
            assert all(len(q['native_call_nanoseconds']) == 3 for q in result['queries'])
            assert all(all(t > 0 for t in q['native_call_nanoseconds']) for q in result['queries'])


def test_rejects_changed_baseline_before_timing(tmp_path):
    specs, queries, refs, config = small_problem(tmp_path)
    baseline = trajectory_core(specs, queries, refs, tmp_path / 'baseline', config,
        kind='epzero', mode='incremental', seed=11, provenance_record={})
    baseline['snapshots'][0]['state']['topology_sha256'] = '0' * 64
    observer = CostObserver(baseline, tmp_path / 'cost')
    with pytest.raises(ValueError, match='did not reproduce'):
        trajectory_core(specs, queries, refs, tmp_path / 'timed', config,
            kind='epzero', mode='incremental', seed=11, provenance_record={}, snapshot_observer=observer)


@pytest.mark.parametrize('kind', ['epzero', 'faiss_hnsw'])
def test_native_counters_and_rankings_repeat(tmp_path, kind):
    from types import SimpleNamespace
    vectors = np.eye(8, dtype=np.float32)
    table = SimpleNamespace(ids=[str(i) for i in range(8)], vectors=vectors)
    backend = Backend(kind, {'M': 4, 'ef_construction': 32}, {'dim': 8, 'metric': 'ip'}, table.ids, 11)
    backend.add(table, table.ids)
    queries = SimpleNamespace(ids=['q'], vectors=np.array([[1., .1, .2, .3, .4, .5, .6, .7]], dtype=np.float32))
    run, record = native_cohort(backend, queries, 4, 16)
    assert run['q'][0][0] == '0'
    # Native counters have different conventions (FAISS can omit entry scoring).
    # Require a populated stable counter, not equality to a common distance total.
    assert record[0]['epzero_distance_computations' if kind == 'epzero' else 'faiss_native_ndis'] > 0
    with pytest.raises(ValueError):
        native_cohort(backend, queries, 4, 16, repeats=1)
