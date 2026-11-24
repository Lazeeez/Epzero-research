import numpy as np
import pytest

from research.longeval import engine_arm_analysis


def table(nq, newest_miss, oldest_miss, rng):
    """[query, cohort, band, (n, missed)] with 20 targets per cohort and band and the given miss rates."""
    t = np.zeros((nq, 3, 3, 2), dtype=np.int64)
    t[..., 0] = 20
    for c, rate in ((0, oldest_miss), (1, oldest_miss), (2, newest_miss)):
        t[:, c, :, 1] = rng.binomial(20, rate, (nq, 3))
    return t


def write(folder, arm, seed, t, prefix=''):
    path = folder / f'{arm}-{seed}'
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / f'{prefix}per-query-ef200.npy', t)


def test_a_clear_repair_passes_and_a_harmful_one_fails(tmp_path):
    rng = np.random.default_rng(0)
    good, bad = tmp_path / 'good', tmp_path / 'bad'
    for seed in (11, 29, 47):
        for folder in (good, bad):
            write(folder, 'grown', seed, table(381, .08, .04, rng))
            write(folder, 'rebuilt', seed, table(381, .045, .045, rng))
        write(good, 'repaired', seed, table(381, .048, .042, rng))
        write(bad, 'repaired', seed, table(381, .048, .07, rng))       # the oldest cohort gets much worse
    out = engine_arm_analysis.analyze(good, draws=500)
    assert out['seeds'] == [11, 29, 47]
    assert out['decision'] == {'effect': True, 'oldest_no_harm': True, 'recall_no_harm': True, 'passed': True}
    assert out['contrasts']['repaired_minus_grown']['newest_excess']['value'] < 0
    harmful = engine_arm_analysis.analyze(bad, draws=500)
    assert harmful['decision']['oldest_no_harm'] is False and harmful['decision']['passed'] is False


def test_arms_must_share_seeds(tmp_path):
    rng = np.random.default_rng(1)
    write(tmp_path, 'grown', 11, table(10, .08, .04, rng))
    write(tmp_path, 'repaired', 11, table(10, .05, .04, rng))
    write(tmp_path, 'rebuilt', 29, table(10, .05, .04, rng))
    with pytest.raises(ValueError):
        engine_arm_analysis.analyze(tmp_path, draws=10)


def test_the_confirmation_reading_is_decided_by_the_effect(tmp_path):
    rng = np.random.default_rng(2)
    for seed in (11, 29, 47):
        write(tmp_path, 'grown', seed, table(100, .08, .04, rng), 'confirm-')
        write(tmp_path, 'repaired', seed, table(100, .048, .042, rng), 'confirm-')
        write(tmp_path, 'rebuilt', seed, table(100, .045, .045, rng), 'confirm-')
    out = engine_arm_analysis.analyze(tmp_path, confirm=True, draws=500)
    assert out['query_count'] == 100 and out['decision']['passed'] == out['decision']['effect']


def test_the_eight_link_arm_is_tested_against_grown_and_other_arms_are_ignored(tmp_path):
    rng = np.random.default_rng(3)
    for seed in (11, 29, 47):
        write(tmp_path, 'grown', seed, table(381, .08, .04, rng))
        write(tmp_path, 'rebuilt', seed, table(381, .045, .045, rng))
        write(tmp_path, 'repaired8', seed, table(381, .046, .043, rng))
    write(tmp_path, 'repaired', 11, table(381, .05, .04, rng))          # another arm, other seeds: ignored
    out = engine_arm_analysis.analyze(tmp_path, draws=500, arm='repaired8')
    assert out['repair'] == 'repaired8' and out['seeds'] == [11, 29, 47]
    assert set(out['arms']) == {'grown', 'repaired8', 'rebuilt'}
    assert out['decision']['passed'] is True and 'repaired8_minus_grown' in out['contrasts']
    with pytest.raises(ValueError):
        engine_arm_analysis.analyze(tmp_path, draws=10)                   # repaired lacks seeds 29 and 47
    with pytest.raises(ValueError):
        engine_arm_analysis.analyze(tmp_path, draws=10, arm='grown')
