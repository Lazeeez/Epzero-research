import sys
import gzip
import json

import numpy as np
import pytest

from research.longeval.retention_search import EventLog, native_parity, replay_search

linux_native = pytest.mark.skipif(sys.platform != 'linux', reason='requires the validated Linux native ABI')


@linux_native
@pytest.mark.parametrize('dimension', [16, 2560])
def test_replay_matches_full_native_trace_results_and_work(dimension):
    import epzero
    rng = np.random.default_rng(86)
    data = rng.normal(size=(150, dimension)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    # Nonsequential labels and insertion order catch accidental label/row substitution.
    index = epzero.Index(dim=dimension, metric='ip', max_elements=150, M=4,
                         ef_construction=32, seed=11)
    index.add(data, labels=rng.permutation(150).astype(np.int64) * 3 + 7, threads=1)
    for query in data[[2, 47]]:
        for effort in (10, 20, 80):
            events = []
            result = native_parity(index, query, 10, effort, emit=events.append)
            assert result['native_parity']
            assert sum(e[0] == 'expand' for e in events) == result['nodes_expanded']
            assert sum(e[0] in ('score', 'entry', 'layer0_entry') for e in events) == result['distance_computations']


def test_deferred_eviction_reaches_hidden_neighbor_and_noop_preserves_baseline():
    graph = {0: [1, 2, 3], 1: [4], 2: [], 3: [], 4: []}
    distances = {0: .5, 1: .6, 2: .4, 3: .3, 4: .01}
    def search(**kwargs):
        return replay_search(lambda node, layer: graph[node], distances.__getitem__,
                             entry=0, entry_layer=0, k=1, ef=2, **kwargs)
    events = []
    base = search(emit=events.append)
    assert base['results'] == [(3, .3)] and base['evicted_unexpanded'] == [1]
    rescued = search(rescue=1)
    assert rescued['results'] == [(4, .01)]
    assert rescued['rescue_triggered'] and rescued['rescue_expanded']
    assert rescued['distance_computations'] == base['distance_computations'] + 1
    for label in (None, 0, 2, 99):
        other = search(rescue=label)
        assert other['results'] == base['results']
        assert other['distance_computations'] == base['distance_computations']
        assert not other['rescue_triggered']
    assert ['expand', 0, 3, False] in events  # Empty-neighbor expansions are explicit.
    bounded = search(rescue=1, distance_budget=base['distance_computations'])
    assert bounded['results'] == base['results'] and bounded['budget_stopped']
    assert not bounded['rescue_expanded']


def test_equal_distances_retain_discovery_order_and_budget_stops_inside_expansion():
    graph = {0: [3, 2, 1], 1: [], 2: [], 3: []}
    distances = {0: 2., 1: 1., 2: 1., 3: 1.}
    def search(**kwargs):
        return replay_search(lambda node, layer: graph[node], distances.__getitem__,
                             entry=0, entry_layer=0, k=2, ef=2, **kwargs)
    result = search()
    assert result['results'] == [(3, 1.), (2, 1.)]
    capped = search(distance_budget=3)
    assert capped['distance_computations'] == 3 and capped['budget_stopped']
    assert capped['results'] == [(3, 1.), (0, 2.)]
    with pytest.raises(RuntimeError, match='distance guard'):
        search(max_distances=3)


def test_complete_event_log_fails_instead_of_truncating(tmp_path):
    log = EventLog(tmp_path/'events.gz', max_events=2)
    log(['entry', 3, .5])
    log(['expand', 0, 3, False])
    with pytest.raises(RuntimeError, match='event-log guard'):
        log(['overflow'])
    receipt = log.close()
    assert receipt['events'] == 2
    with gzip.open(tmp_path/'events.gz', 'rt') as stream:
        assert [json.loads(line) for line in stream] == [['entry', 3, .5], ['expand', 0, 3, False]]


@linux_native
def test_corrupted_native_trace_is_rejected(monkeypatch):
    import epzero
    index = epzero.Index(dim=2, metric='ip', max_elements=3, M=2, ef_construction=4)
    index.add(np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32), threads=1)
    original = epzero.Index.trace
    def corrupt(self, *args, **kwargs):
        trace = original(self, *args, **kwargs)
        trace['nodes_expanded'] += 1
        return trace
    monkeypatch.setattr(epzero.Index, 'trace', corrupt)
    with pytest.raises(ValueError, match='nodes_expanded parity'):
        native_parity(index, np.array([1., 0.], dtype=np.float32), 1, 2)
