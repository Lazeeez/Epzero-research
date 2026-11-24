import numpy as np
import pytest

from research.longeval import calibration_screen


@pytest.mark.parametrize('metric', ['ip', 'l2'])
def test_exact_neighbours_among_the_present_documents(metric):
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((900, 12)).astype(np.float32)
    queries = rng.standard_normal((7, 12)).astype(np.float32)
    present = rng.random(900) < 0.6
    got = calibration_screen.exact_among(vectors, queries, present, metric, k=100, chunk=128)
    ids = np.flatnonzero(present)
    v, q = vectors[ids].astype(np.float64), queries.astype(np.float64)
    scores = q @ v.T if metric == 'ip' else -((q[:, None] - v[None]) ** 2).sum(-1)
    for row in range(len(q)):
        expected = ids[np.lexsort((ids, -scores[row]))[:100]]
        assert np.array_equal(got[row], expected)
    assert present[got].all()


def test_the_query_split_is_fixed_and_disjoint():
    cal, ev = calibration_screen.split(381)
    again = calibration_screen.split(381)
    assert np.array_equal(cal, again[0]) and np.array_equal(ev, again[1])
    assert len(cal) == 100 and len(ev) == 281 and not set(cal) & set(ev)


def test_the_table_counts_misses_by_cohort():
    cohort = np.array([1, 2, 3] * 40, dtype=np.int8)
    targets = np.arange(100)[None]
    found = targets.copy()
    found[0, 5] = -1                                       # target 5 (cohort 3, rank 6) missed
    table = calibration_screen.table_of(found, targets, cohort)
    assert table[..., 0].sum() == 100 and table[0, 2, :, 1].sum() == 1 and table[..., 1].sum() == 1
