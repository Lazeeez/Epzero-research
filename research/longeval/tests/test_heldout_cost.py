import copy

import pytest

from research.longeval import trajectory
from research.longeval.heldout_cost import HeldoutCostObserver, summarize
from research.longeval.tests.test_heldout import as_evaluation
from research.longeval.tests.test_trajectory import small_problem


@pytest.mark.parametrize('kind', ['epzero', 'faiss_hnsw'])
def test_timing_reproduces_the_heldout_baseline_and_records_work(tmp_path, monkeypatch, kind):
    specs, queries, refs, config = small_problem(tmp_path)
    queries = as_evaluation(queries)
    baseline = trajectory.trajectory_core(specs, queries, refs, tmp_path / 'base', config, kind=kind, mode='incremental',
                                          seed=11, provenance_record={}, evaluation_role='evaluation')
    observer = HeldoutCostObserver(baseline, tmp_path / 'cost/native-cost')
    monkeypatch.setattr('research.longeval.heldout_cost.WIDE_EF', 40)  # the fixture has 48 documents
    report = trajectory.trajectory_core(specs, queries, refs, tmp_path / 'cost/trajectory', config, kind=kind,
                                        mode='incremental', seed=11, provenance_record={}, snapshot_observer=observer,
                                        evaluation_role='evaluation')
    assert all(s['diagnostic']['baseline_reproduced'] for s in report['snapshots'])
    rows = summarize(tmp_path / 'cost', tmp_path / 'summary')
    assert len(rows) == 9 and {r['regime'] for r in rows} == {'frozen', 'matched_recall', 'wide'}
    assert all(r['queries'] == 6 and r['work_mean'] > 0 and r['ms_mean'] > 0 for r in rows)
    changed = copy.deepcopy(baseline)
    changed['snapshots'][1]['runs']['frozen']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='did not reproduce'):
        trajectory.trajectory_core(specs, queries, refs, tmp_path / 'bad/trajectory', config, kind=kind,
                                   mode='incremental', seed=11, provenance_record={},
                                   snapshot_observer=HeldoutCostObserver(changed, tmp_path / 'bad/native-cost'),
                                   evaluation_role='evaluation')


def test_observer_rejects_a_development_baseline(tmp_path):
    with pytest.raises(ValueError, match='held-out incremental'):
        HeldoutCostObserver({'status': 'completed', 'mode': 'incremental', 'backend': 'epzero'}, tmp_path / 'x')
