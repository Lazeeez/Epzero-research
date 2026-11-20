import sys
import copy
import gzip
import json

import pytest

from research.longeval.contracts import digest
from research.longeval.retention_diagnostic import RetentionObserver
from research.longeval.retention_search import replay_search
from research.longeval.tests.test_trajectory import small_problem
from research.longeval.trajectory import trajectory_core

linux_native = pytest.mark.skipif(sys.platform != 'linux', reason='requires the validated Linux native ABI')


def replay_saved_log(path, arm, k):
    with gzip.open(path, 'rt') as stream:
        events = [json.loads(line) for line in stream]
    adjacency, distances = {}, {}
    for event in events:
        if event[0] == 'neighbors':
            _, layer, origin, other = event
            key = (origin, layer)
            assert key not in adjacency or adjacency[key] == other
            adjacency[key] = other
        if event[0] in ('entry', 'layer0_entry'):
            label, distance = event[1:]
        elif event[0] == 'score':
            label, distance = event[3:5]
        else:
            continue
        assert label not in distances or distances[label] == distance
        distances[label] = distance
    observed = []
    result = replay_search(lambda node, layer: adjacency[node, layer], distances.__getitem__,
        entry=arm['entry_label'], entry_layer=arm['entry_layer'], k=k, ef=arm['ef'],
        rescue=arm['rescue_label'], distance_budget=arm['distance_budget'], emit=observed.append)
    assert observed == events
    for key in ('distance_computations', 'nodes_expanded', 'rescue_triggered', 'rescue_expanded', 'budget_stopped'):
        assert result[key] == arm[key]
    assert digest(result['evicted_unexpanded']) == arm['evicted_unexpanded_sha256']


@linux_native
def test_observer_reproduces_trajectory_and_exports_replayable_complete_interventions(tmp_path):
    specs, queries, refs, config = small_problem(tmp_path)
    baseline = trajectory_core(specs, queries, refs, tmp_path/'baseline', config,
                              kind='epzero', mode='incremental', seed=11, provenance_record={})
    observer = RetentionObserver(json.loads(json.dumps(baseline)), tmp_path/'retention', tmp_path/'work',
        targets=queries['development'].ids[:2], bridges=('d000', 'd001'),
        low_efforts=(10, 20), baseline_efforts=(10, 11, 20, 21, 40))
    result = trajectory_core(specs, queries, refs, tmp_path/'reproduced', config,
        kind='epzero', mode='incremental', seed=11, provenance_record={}, snapshot_observer=observer)
    for a, b in zip(baseline['snapshots'], result['snapshots']):
        assert a['state'] == b['state'] and a['runs'] == b['runs']
    assert not (tmp_path/'retention/snapshot-1').exists()
    folder = tmp_path/'retention/snapshot-3'
    report = json.loads((folder/'report.json').read_text())
    assert report['status'] == 'completed' and report['state_unchanged']
    assert len(report['arms']) == 68
    for arm in report['arms']:
        assert arm['event_log']['complete']
        replay_saved_log(folder/arm['event_log']['file'], arm, config['k'])
        if arm['arm'] == 'wide_beam_distance_cap':
            assert arm['distance_computations'] <= arm['distance_budget']
            paired = next(row for row in report['arms'] if row['name'] == arm['paired_intervention'])
            assert arm['distance_budget'] == paired['distance_computations']


@pytest.mark.parametrize('corruption', ['state', 'ranking'])
def test_bad_baseline_cannot_start_intervention(tmp_path, corruption):
    specs, queries, refs, config = small_problem(tmp_path)
    baseline = trajectory_core(specs, queries, refs, tmp_path/'baseline', config,
                              kind='epzero', mode='incremental', seed=11, provenance_record={})
    baseline = copy.deepcopy(baseline)
    if corruption == 'state':
        baseline['snapshots'][0]['state']['topology_sha256'] = '0' * 64
    else:
        baseline['snapshots'][0]['runs']['frozen']['sha256'] = '0' * 64
    observer = RetentionObserver(baseline, tmp_path/'retention', tmp_path/'work',
        targets=queries['development'].ids[:2], bridges=('d000', 'd001'),
        low_efforts=(10, 20), baseline_efforts=(10, 11, 20, 21, 40))
    with pytest.raises(ValueError, match='reproduction failed'):
        trajectory_core(specs, queries, refs, tmp_path/'reproduced', config,
            kind='epzero', mode='incremental', seed=11, provenance_record={}, snapshot_observer=observer)
    assert not list((tmp_path/'retention').iterdir())
