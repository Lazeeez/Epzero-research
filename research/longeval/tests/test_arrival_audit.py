import numpy as np
import pytest

from research.longeval import arrival_audit


def synthetic(queries=60, n=400, seed=1):
    rng = np.random.default_rng(seed)
    cohort = np.repeat(np.array([1, 2, 3], dtype=np.int8), [n // 2, n // 4, n - n // 2 - n // 4])
    exact = np.stack([rng.choice(n, 100, replace=False) for _ in range(queries)])
    builds = {}
    for name in ('epzero/appended-11', 'epzero/reversed-11', 'faiss_hnsw/appended-11', 'faiss_hnsw/reversed-11'):
        labels = exact.copy()
        for q in range(queries):
            # each build drops two of the exact top 10 and promotes two from ranks 11-100
            drop = rng.choice(10, 2, replace=False)
            labels[q, drop] = exact[q, 10 + rng.choice(90, 2, replace=False)]
        builds[name] = labels
    doc_ids = [f'd{i}' for i in range(n)]
    query_ids = [f'q{i}' for i in range(queries)]
    return builds, exact, cohort, doc_ids, query_ids


def test_roles_and_selection_are_balanced_blind_and_reproducible():
    builds, exact, cohort, doc_ids, query_ids = synthetic()
    dropped, replacements, both = arrival_audit.build_roles(builds, exact, cohort, doc_ids, query_ids, query_ids)
    assert all(item['group'] in (1, 3) for item in dropped.values())
    chosen, counts = arrival_audit.select(dropped, replacements, both, per_group=10, overlap=3)
    again, _ = arrival_audit.select(dropped, replacements, both, per_group=10, overlap=3)
    assert chosen == again
    roles = [(c['role'], c['arrival_group']) for c in chosen]
    assert roles.count(('dropped', 3)) == roles.count(('dropped', 1)) == 10
    assert roles.count(('replacement', 3)) == roles.count(('replacement', 1)) == 10
    assert sum(c['overlap'] for c in chosen) == 2 * 2 * 3
    pairs = [(c['query_id'], c['doc_id']) for c in chosen]
    assert len(pairs) == len(set(pairs)) and not set(pairs) & both
    for drop, rep in zip(chosen[::2], chosen[1::2]):
        assert (drop['role'], rep['role']) == ('dropped', 'replacement')
        assert drop['query_id'] == rep['query_id'] and drop['build'] == rep['build']
        assert rep['doc_id'] in replacements[(rep['build'], rep['query_id'])]


def test_selection_refuses_too_few_pairs():
    builds, exact, cohort, doc_ids, query_ids = synthetic(queries=3)
    dropped, replacements, both = arrival_audit.build_roles(builds, exact, cohort, doc_ids, query_ids, query_ids)
    with pytest.raises(ValueError):
        arrival_audit.select(dropped, replacements, both, per_group=50)


def test_a_balanced_census_when_a_group_is_short():
    builds, exact, cohort, doc_ids, query_ids = synthetic(queries=12)
    dropped, replacements, both = arrival_audit.build_roles(builds, exact, cohort, doc_ids, query_ids, query_ids)
    newest = sum(1 for k, v in dropped.items() if v['group'] == 3 and k not in both)
    chosen, counts = arrival_audit.select(dropped, replacements, both, per_group=10_000, overlap=2, census=True)
    smaller = min(newest, sum(1 for k, v in dropped.items() if v['group'] == 1 and k not in both))
    assert counts['balanced_census'] == smaller
    roles = [(c['role'], c['arrival_group']) for c in chosen]
    assert roles.count(('dropped', 3)) == roles.count(('dropped', 1)) == smaller
