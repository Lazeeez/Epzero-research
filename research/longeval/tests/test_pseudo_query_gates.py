import numpy as np
import pytest

from research.longeval import pseudo_query_gates as g


def test_components_on_induced_subgraph_only():
    out = [[1], [2], [], [4], [], [0]]          # 0-1-2 chain, 3-4, 5->0
    adj = g.undirected(out)
    assert g.components([0, 2, 3, 4], adj) == [[0], [2], [3, 4]]   # 1 is outside the set
    assert sorted(map(sorted, g.components([0, 1, 2, 5], adj))) == [[0, 1, 2, 5]]


def test_bridge_joins_components_to_anchor_and_respects_budget():
    base = [[1], [0], [3], [2], [], []]
    added = g.bridge(base, [[0, 2, 4, 5, 1]], budget=2)
    # Components of {0,2,4,5,1}: {0,1} anchor, {2}, {4}, {5}; anchor budget 2 limits joins.
    assert added[0] == [2, 4] and added[2] == [0] and added[4] == [0] and added[5] == []
    assert all(len(a) <= 2 for a in added)
    # Already-joined answers add nothing on a repeat.
    again = g.bridge([b + a for b, a in zip(base, added)], [[0, 2, 4]], budget=2)
    assert sum(map(len, again)) == 0


def test_random_bridge_matches_edge_count_and_budget():
    base = [[] for _ in range(50)]
    added = g.random_bridge(base, 40, budget=3, seed=7)
    assert sum(map(len, added)) == 80
    assert all(len(a) <= 3 and len(set(a)) == len(a) for a in added)
    assert all(u in added[v] for u in range(50) for v in added[u])
    assert added == g.random_bridge(base, 40, budget=3, seed=7)


def test_equal_work_gain_interpolates_and_refuses_extrapolation():
    curve = [(100, 0.80), (200, 0.90), (400, 0.95)]
    assert g.equal_work_gain(curve, 150, 0.88) == pytest.approx(0.03)
    assert g.equal_work_gain(curve, 400, 0.95) == pytest.approx(0.0)
    assert g.equal_work_gain(curve, 50, 0.9) is None


def test_exact_topk_orders_by_score_then_position():
    data = np.array([[1, 0], [0, 1], [1, 0], [0.6, 0.8]], dtype=np.float32)
    q = np.array([[1, 0], [0, 1]], dtype=np.float32)
    assert g.exact_topk(q, data, 3).tolist() == [[0, 2, 3], [1, 3, 0]]
