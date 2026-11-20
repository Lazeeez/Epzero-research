import sys
import numpy as np
import pytest

from research.longeval.frontier_search import frontier_search
from research.longeval.retention_search import replay_search, native_parity, NativeScorer

linux_native = pytest.mark.skipif(sys.platform != 'linux', reason='requires the validated Linux native ABI')


def fixture(policy, distances=None, **kwargs):
    graph = {0: [1, 2, 3, 4], 1: [5], 2: [0, 3], 3: [0, 2],
             4: [0, 2, 3, 6], 5: [], 6: []}
    distances = distances or {0: .5, 1: .6, 2: .4, 3: .3, 4: .45, 5: .01, 6: .7}
    calls, events, reads = [], [], []
    def score(label):
        calls.append(label)
        return distances[label]
    def neighbors(label, layer):
        reads.append(label)
        return graph[label]
    result = frontier_search(neighbors, score, entry=0, entry_layer=0, k=1, ef=2,
                             policy=policy, emit=events.append, **kwargs)
    assert result['distance_computations'] == len(calls)
    assert result['adjacency_reads'] == len(reads)
    assert result['adjacency_ids_inspected'] == sum(len(graph[v]) for v in reads)
    assert len(calls[2:]) == len(set(calls[2:]))  # Upper and layer-0 entry are repeated.
    return result, calls, events


def test_online_policy_can_change_route_at_equal_distance_work():
    overlap, calls, events = fixture('overlap', distance_budget=7, max_resumptions=1)
    closest, _, _ = fixture('closest', distance_budget=7, max_resumptions=1)
    assert overlap['distance_computations'] == closest['distance_computations'] == 7
    assert overlap['results'] == [(5, .01)] and closest['results'] == [(3, .3)]
    assert overlap['selections'][0]['label'] == 1
    # Lookahead at 4 must not score or mark its unseen neighbor 6 as visited.
    assert 6 not in calls and overlap['adjacency_reads'] > closest['adjacency_reads']
    assert sum(e[0] == 'expand' for e in events) == overlap['nodes_expanded']


def test_unseen_fraction_is_not_a_recall_guarantee():
    distances = {0: .5, 1: .6, 2: .4, 3: .3, 4: .45, 5: .7, 6: .01}
    overlap, _, _ = fixture('overlap', distances, distance_budget=7, max_resumptions=1)
    closest, _, _ = fixture('closest', distances, distance_budget=7, max_resumptions=1)
    assert overlap['results'] == [(3, .3)] and closest['results'] == [(6, .01)]


@pytest.mark.parametrize('capacity', [0, 1, 2, 8])
def test_reserve_bound_and_exhaustive_tiny_graph(capacity):
    result, _, _ = fixture('overlap', reserve_capacity=capacity, max_resumptions=20)
    assert result['reserve_peak'] <= capacity
    assert result['lookahead_ids_peak'] <= 5
    if capacity >= 2:
        assert result['results'] == [(5, .01)]
        assert result['distance_computations'] == 8


def test_noop_and_mid_expansion_budget_match_original():
    graph = {0: [3, 2, 1], 1: [], 2: [], 3: []}
    distances = {0: 2., 1: 1., 2: 1., 3: 1.}
    for cap in (None, 3, 4, 5):
        args = dict(entry=0, entry_layer=0, k=2, ef=2, distance_budget=cap)
        expected = replay_search(lambda v, _: graph[v], distances.__getitem__, **args)
        for policy, extra in [('off', {}), ('closest', {'max_resumptions': 0}),
                              ('overlap', {'reserve_capacity': 0})]:
            actual = frontier_search(lambda v, _: graph[v], distances.__getitem__,
                                     policy=policy, **args, **extra)
            for key in ('results', 'distance_computations', 'nodes_expanded', 'upper_events'):
                assert actual[key] == expected[key]


def test_budget_and_nonfinite_fail_closed():
    with pytest.raises(ValueError, match='configuration'):
        fixture('overlap', distance_budget=1)
    with pytest.raises(ValueError, match='nonfinite'):
        fixture('overlap', {0: float('nan')})
    with pytest.raises(RuntimeError, match='guard'):
        fixture('overlap', max_distances=3)
    with pytest.raises(ValueError, match='upper-layer route'):
        frontier_search(lambda v, _: [1, 2], lambda v: 3 - v,
                        entry=0, entry_layer=1, k=1, ef=2, distance_budget=2)


@linux_native
@pytest.mark.parametrize('dimension', [16, 2560])
def test_off_matches_native_and_resumption_preserves_graph(dimension):
    import epzero
    rng = np.random.default_rng(20261118)
    data = rng.normal(size=(180, dimension)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    labels = rng.permutation(180).astype(np.int64) * 3 + 7
    index = epzero.Index(dim=dimension, metric='ip', max_elements=180, M=4,
                         ef_construction=32, seed=11)
    index.add(data, labels=labels, threads=1)
    before = {int(label): tuple(index.neighbors(int(label), 0)) for label in labels}
    for query in data[[2, 47]]:
        for ef in (10, 20, 80):
            baseline = native_parity(index, query, 10, ef)
            events = []
            args = dict(entry=baseline['entry_label'], entry_layer=baseline['entry_layer'],
                        k=10, ef=ef)
            actual = frontier_search(index.neighbors, NativeScorer(index, query),
                                     policy='off', emit=events.append, **args)
            for key in ('results', 'distance_computations', 'nodes_expanded', 'upper_events'):
                assert actual[key] == baseline[key]
            assert [tuple(e[1:]) for e in events if e[0] == 'score'] == index.trace(query, k=10, ef=ef)['events']
            capped = frontier_search(index.neighbors, NativeScorer(index, query),
                                     distance_budget=200, policy='overlap', **args)
            assert capped['distance_computations'] <= 200
            assert capped['reserve_peak'] <= 256
    assert before == {int(label): tuple(index.neighbors(int(label), 0)) for label in labels}
