import sys
import copy
import json

import numpy as np
import pytest

from research.longeval.retention_search import NativeScorer, replay_search
from research.longeval.tests.test_trajectory import small_problem
from research.longeval.trajectory import trajectory_core
from research.longeval.work_budget import WorkBudgetObserver, prefix_searches, select_budget

linux_native = pytest.mark.skipif(sys.platform != 'linux', reason='requires the validated Linux native ABI')


@linux_native
@pytest.mark.parametrize('dimension', [16, 2560])
def test_native_prefixes_equal_explicitly_budget_stopped_search(dimension):
    import epzero
    rng = np.random.default_rng(97)
    data = rng.normal(size=(180, dimension)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    index = epzero.Index(dim=dimension, metric='ip', max_elements=180, M=4, ef_construction=32)
    index.add(data, labels=rng.permutation(180).astype(np.int64)*3+5, threads=1)
    for query in data[[7, 71]]:
        trace = index.trace(query, k=10, ef=80)
        scorer = NativeScorer(index, query)
        start = trace['entry_label']
        for level, source, target, distance, accepted in trace['events']:
            if level > 0 and accepted:
                start = target
        prefixes = prefix_searches(trace, scorer(start), (1, 2, 4, 8, 16, 32, 64, 128, 256, 512), 10)
        assert any(not p['feasible'] for p in prefixes.values())
        assert prefixes[512]['results'] == trace['results']
        for budget, prefix in prefixes.items():
            if not prefix['feasible']:
                continue
            stopped = replay_search(index.neighbors, scorer, entry=trace['entry_label'],
                entry_layer=trace['entry_layer'], k=10, ef=80, distance_budget=budget)
            assert stopped['results'] == prefix['results']
            assert stopped['distance_computations'] == prefix['distance_computations']


def test_prefixes_handle_ties_exhaustion_and_reject_broken_accounting():
    trace = {'entry_label': 0, 'events': [(0, 0, 3, 1., True), (0, 0, 2, 1., True), (0, 0, 1, 1., False)],
             'distance_computations': 5, 'results': [(3, 1.), (2, 1.)]}
    values = prefix_searches(trace, 2., (1, 2, 3, 4, 5, 8), 2)
    assert not values[1]['feasible'] and not values[2]['feasible']
    assert values[3]['results'] == [(3, 1.), (0, 2.)]
    assert values[8]['results'] == values[5]['results'] == trace['results']
    assert values[8]['distance_computations'] == 5 and values[8]['full_native_score_stream_covered']
    broken = copy.deepcopy(trace)
    broken['distance_computations'] += 1
    with pytest.raises(ValueError, match='accounting'):
        prefix_searches(broken, 2., (8,), 2)
    broken = copy.deepcopy(trace)
    broken['results'].reverse()
    with pytest.raises(ValueError, match='ranking'):
        prefix_searches(broken, 2., (8,), 2)


def test_budget_selection_does_not_drop_infeasible_queries_or_assume_monotonicity():
    points = [{'budget': 10, 'feasible': False, 'mean_recall': None},
              {'budget': 20, 'feasible': True, 'mean_recall': .96},
              {'budget': 40, 'feasible': True, 'mean_recall': .94}]
    assert select_budget(points, .95) == 20
    assert select_budget(points, .99) is None


@linux_native
def test_full_cohort_observer_preserves_baseline_and_freezes_snapshot1_budget(tmp_path):
    specs, queries, refs, config = small_problem(tmp_path)
    baseline = trajectory_core(specs, queries, refs, tmp_path/'baseline', config,
        kind='epzero', mode='incremental', seed=47, provenance_record={})
    observer = WorkBudgetObserver(json.loads(json.dumps(baseline)), tmp_path/'work-budget', tmp_path/'work',
        budgets=(1, 16, 32, 64, 128), wide_ef=40, targets=queries['development'].ids[:2])
    repeated = trajectory_core(specs, queries, refs, tmp_path/'repeated', config,
        kind='epzero', mode='incremental', seed=47, provenance_record={}, snapshot_observer=observer)
    frozen = None
    for step, (old, new) in enumerate(zip(baseline['snapshots'], repeated['snapshots']), 1):
        assert old['state'] == new['state'] and old['runs'] == new['runs']
        report = json.loads((tmp_path/f'work-budget/snapshot-{step}/report.json').read_text())
        assert report['status'] == 'completed' and report['state_unchanged']
        assert len(report['queries']) == 10 and len(report['runs']) == 14
        assert len({row['query_id'] for row in report['queries']}) == 10
        assert sum('explicit_stopping_validation' in row for row in report['queries']) == 4
        assert all(not report['cohorts'][role][0]['feasible'] for role in ('development', 'calibration'))
        chosen = select_budget(report['cohorts']['calibration'], .95)
        assert report['selection']['current_calibration_budget'] == chosen
        if step == 1:
            frozen = chosen
        assert report['selection']['frozen_snapshot1_budget'] == frozen
        for row in report['queries']:
            assert all(trace['events_truncated'] == 0 for trace in row['native'].values())


@pytest.mark.parametrize('corruption', ['state', 'ranking'])
def test_invalid_baseline_blocks_work_budget_observation(tmp_path, corruption):
    specs, queries, refs, config = small_problem(tmp_path)
    baseline = trajectory_core(specs, queries, refs, tmp_path/'baseline', config,
        kind='epzero', mode='incremental', seed=47, provenance_record={})
    if corruption == 'state':
        baseline['snapshots'][0]['state']['topology_sha256'] = '0'*64
    else:
        baseline['snapshots'][0]['runs']['frozen']['sha256'] = '0'*64
    observer = WorkBudgetObserver(baseline, tmp_path/'work-budget', tmp_path/'work',
        budgets=(16, 32, 64), wide_ef=40, targets=queries['development'].ids[:2])
    with pytest.raises(ValueError, match='reproduction failed'):
        trajectory_core(specs, queries, refs, tmp_path/'repeated', config,
            kind='epzero', mode='incremental', seed=47, provenance_record={}, snapshot_observer=observer)
    assert not list((tmp_path/'work-budget').iterdir())
