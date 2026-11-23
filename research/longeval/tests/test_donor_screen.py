import numpy as np

from research.longeval import donor_screen


def toy(n=12, width=3, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, 4)).astype(np.float32)
    lists = np.full(n * width, -1, dtype=np.int32)
    for s in range(n):
        others = [x for x in rng.permutation(n) if x != s][:width - 1]     # one free slot per node
        lists[s * width:s * width + len(others)] = others
    rows = np.arange(n, dtype=np.int64) * width
    return vectors, lists, rows, width


def test_intervention_respects_caps_and_counts_every_change():
    vectors, lists, rows, width = toy()
    before = lists.copy()
    sample_of = np.arange(len(rows))
    targets = np.array([0, 1, 2])
    donors_of = {t: [d for d in range(3, 12)] for t in (0, 1, 2)}
    counts = donor_screen.intervene(lists, rows, width, vectors, sample_of, targets, donors_of, 'closest',
                                    np.random.default_rng(1))
    changed = int((lists != before).sum())
    assert counts['links_added'] == changed
    for t in targets:
        new_links = sum(t in lists[r:r + width] and t not in before[r:r + width] for r in rows)
        assert new_links <= donor_screen.PER_TARGET
    for d in range(len(rows)):
        gained = sum(x not in before[rows[d]:rows[d] + width] for x in lists[rows[d]:rows[d] + width] if x >= 0)
        assert gained <= donor_screen.DONOR_CAP
    # Every added link either filled a free slot or displaced exactly one existing link.
    filled = int(((before < 0) & (lists >= 0)).sum())
    assert counts['links_displaced'] == counts['links_added'] - filled


def test_arms_differ_only_in_donor_order():
    vectors, lists, rows, width = toy(seed=3)
    targets = np.array([0, 1])
    donors_of = {0: list(range(2, 12)), 1: list(range(2, 12))}
    totals = []
    for arm in ('closest', 'uniform', 'hub'):
        copy = lists.copy()
        counts = donor_screen.intervene(copy, rows, width, vectors, np.arange(len(rows)), targets, donors_of, arm,
                                        np.random.default_rng(2))
        totals.append(counts['links_added'])
    assert len(set(totals)) == 1           # same targets, eligible sets and caps: the same number of links
