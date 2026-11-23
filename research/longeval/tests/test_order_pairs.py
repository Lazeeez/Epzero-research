import numpy as np
import pytest

from research.longeval import order_pairs


def test_transitions_count_each_pair_once():
    first = np.array([[True, False, True, False]])
    second = np.array([[True, True, False, False]])
    cohort = np.array([[1, 1, 3, 3]])
    t = order_pairs.transitions(first, second, cohort)
    # cohort 1: target 0 missed both, target 1 newly lost; cohort 3: target 2 rescued, target 3 found both
    assert t[0, 0].tolist() == [0, 1, 0, 1]
    assert t[0, 2].tolist() == [1, 0, 1, 0]
    assert t.sum() == 4


def test_compare_recovers_a_known_paired_change(tmp_path):
    rng = np.random.default_rng(3)
    queries, n = 40, 1000
    targets = np.stack([rng.choice(n, 100, replace=False) for _ in range(queries)]).astype(np.int32)
    cohort = np.repeat(np.array([1, 3], dtype=np.int8), n // 2)
    sample = tmp_path / 'sample'
    sample.mkdir()
    np.save(sample / 'cohort.npy', cohort)
    np.save(sample / 'exact-top100.npy', targets)
    builds = tmp_path / 'builds'

    def save(order, seed, drop):
        # Return every target except those `drop` selects.
        labels = np.where(drop, -1, targets).astype(np.int32)
        folder = builds / f'{order}-{seed}'
        folder.mkdir(parents=True)
        np.save(folder / 'labels-ef200.npy', labels)
    newest = cohort[targets] == 3
    dropped = newest & (np.arange(100) % 10 == 0)                    # appended misses these newest-cohort targets
    for s in (11, 29):
        save('appended', s, dropped)
        save('reversed', s, np.zeros_like(newest))                   # reversed misses nothing
    out = order_pairs.compare(sample, builds, 'appended', 'reversed')
    assert out['seeds'] == [11, 29]
    newest_block = out['cohorts']['3']
    expected = -100 * dropped.sum() / newest.sum()
    assert newest_block['newly_lost'] == 0 and newest_block['rescued'] == dropped.sum()
    assert newest_block['risk_change_pp'] == pytest.approx(expected)
    assert out['cohorts']['1']['risk_change_pp'] == 0
    assert out['gap_change_pp']['value'] == pytest.approx(expected)


def test_compare_refuses_orders_without_a_shared_seed(tmp_path):
    sample = tmp_path / 'sample'
    sample.mkdir()
    np.save(sample / 'cohort.npy', np.ones(10, dtype=np.int8))
    np.save(sample / 'exact-top100.npy', np.zeros((1, 100), dtype=np.int32))
    (tmp_path / 'builds' / 'appended-11').mkdir(parents=True)
    with pytest.raises(ValueError):
        order_pairs.compare(sample, tmp_path / 'builds', 'appended', 'reversed')
