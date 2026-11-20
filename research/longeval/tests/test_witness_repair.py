import pytest

from research.longeval.retention_search import replay_search
from research.longeval.witness_repair import Trace, Overlay, propose, select


def problem():
    graph = {0: [1, 2, 3], 1: [4], 2: [], 3: [], 4: [5], 5: []}
    distances = {0: .5, 1: .6, 2: .4, 3: .3, 4: .2, 5: .01}
    low, high = Trace(), Trace()
    def run(ef, emit, neighbors=None):
        return replay_search(neighbors or (lambda v, layer: graph[v]), distances.__getitem__,
                             entry=0, entry_layer=0, k=1, ef=ef, emit=emit)
    result = run(2, low)
    run(6, high)
    return graph, low, high, result, run


def test_witness_proposes_gateway_before_answer_and_repair_is_query_blind():
    graph, low, high, result, run = problem()
    assert high.path(5) == [0, 1, 4, 5]
    row = propose('train', low, high, result, [5])
    assert (row['source'], row['destination'], row['endpoint']) == (0, 4, 5)
    assert not row['destination_in_exact'] and row['skipped_nodes'] == 1
    chosen = select([row], {'train'})
    overlay = Overlay(lambda v, layer: graph[v], [(r['source'], r['destination']) for r in chosen], graph)
    assert run(2, lambda e: None, overlay)['results'] == [(5, .01)]
    assert graph[0] == [1, 2, 3] and overlay.edge_reads > 0
    assert overlay(0, 1) == graph[0]  # Only layer zero changes.


def test_fit_rejects_development_and_duplicate_queries():
    _, low, high, result, _ = problem()
    row = propose('dev', low, high, result, [5])
    with pytest.raises(ValueError, match='non-calibration'):
        select([row], {'train'})
    with pytest.raises(ValueError, match='one proposal'):
        select([row, row], {'dev'})


def test_edge_budget_support_and_deterministic_order():
    proposals = [dict(query_id=str(q), source=s, destination=d, skipped_nodes=skip)
                 for q, s, d, skip in [(0, 1, 2, 2), (1, 1, 3, 9), (2, 1, 2, 2), (3, 4, 5, 1)]]
    chosen = select(proposals, {'0', '1', '2', '3'}, edge_budget=2, per_source=1)
    assert [(r['source'], r['destination']) for r in chosen] == [(1, 2), (4, 5)]
    assert chosen == select(list(reversed(proposals)), {'0', '1', '2', '3'}, edge_budget=2, per_source=1)
    assert select(proposals, {'0', '1', '2', '3'}, edge_budget=0) == []


@pytest.mark.parametrize('edges', [[(0, 0)], [(0, 99)], [(0, 1)], [(0, 4), (0, 4)], [(2, 0), (2, 1), (2, 3)]])
def test_invalid_overlay_rejected(edges):
    graph, *_ = problem()
    with pytest.raises(ValueError):
        Overlay(lambda v, layer: graph[v], edges, graph)


def test_noop_and_trace_corruption():
    graph, low, high, result, run = problem()
    overlay = Overlay(lambda v, layer: graph[v], [], graph)
    assert run(2, lambda e: None, overlay)['results'] == result['results']
    assert propose('train', low, high, result, [3]) is None
    high.adjacency[1] = ()
    with pytest.raises(ValueError, match='unwitnessed'):
        high.path(5)
