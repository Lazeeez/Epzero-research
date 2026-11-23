"""Within-query miss ratios: the stratified estimators and the aggregate over collected builds."""
import json

import numpy as np
import pytest

from research.longeval import insertion_strata as strata
from research.longeval.insertion_pilot import BANDS, _ratio


def table(rows):
    """rows: per query, {(cohort, band): (neighbours, missed)} -> [query, cohort, band, 2]."""
    out = np.zeros((len(rows), 3, len(BANDS), 2), dtype=np.int64)
    for q, cells in enumerate(rows):
        for (cohort, band), pair in cells.items():
            out[q, cohort - 1, band] = pair
    return out


def test_one_stratum_gives_the_plain_ratio_of_rates():
    one = table([{(1, 0): (10, 1), (3, 0): (20, 6)}])
    assert strata.mantel_haenszel(one, 3) == pytest.approx(.3 / .1)
    assert strata.risk_difference(one, 3) == pytest.approx(.3 - .1)
    assert _ratio(one.sum(axis=0), 3, 1) == pytest.approx(.3 / .1)


def test_a_hard_query_with_many_later_neighbours_inflates_only_the_pooled_ratio():
    # In each query both groups have the same miss rate; the hard query holds most of the later neighbours.
    confounded = table([{(1, 0): (90, 9), (3, 0): (10, 1)},        # easy query: 10% missed in both groups
                        {(1, 0): (10, 5), (3, 0): (90, 45)}])      # hard query: 50% missed in both groups
    assert _ratio(confounded.sum(axis=0), 3, 1) == pytest.approx((46 / 100) / (14 / 100))
    assert strata.mantel_haenszel(confounded, 3) == pytest.approx(1.0)
    assert strata.risk_difference(confounded, 3) == pytest.approx(0.0)


def test_strata_without_one_group_carry_no_weight_and_groups_can_be_pooled():
    sparse = table([{(1, 0): (5, 1)},                              # no later neighbour: ignored
                    {(3, 1): (4, 2)},                              # no reference neighbour: ignored
                    {(1, 2): (8, 2), (2, 2): (4, 2), (3, 2): (4, 2)}])
    assert strata.mantel_haenszel(sparse, (2, 3)) == pytest.approx(.5 / .25)
    assert strata.mantel_haenszel(sparse, 3) == pytest.approx(.5 / .25)
    assert np.isnan(strata.mantel_haenszel(table([{(3, 0): (4, 2)}]), 3))


def write_builds(root, backend, *, queries=40, hard_share=.5):
    """Three orders x two seeds. Hard queries hold most later neighbours; within a query, the order alone moves misses."""
    rng = np.random.default_rng(7)
    for order, factor in (('appended', 1.5), ('shuffled', 1.0), ('reversed', 1 / 1.5)):
        for seed in (11, 29):
            rows = []
            for q in range(queries):
                hard = q < queries * hard_share
                base = .30 if hard else .06
                later = 60 if hard else 20
                cells = {}
                for band in range(len(BANDS)):
                    cells[(1, band)] = (100 - later, rng.binomial(100 - later, base))
                    cells[(3, band)] = (later, rng.binomial(later, min(1, base * factor)))
                rows.append(cells)
            folder = root / backend / f'{order}-{seed}'
            folder.mkdir(parents=True)
            np.save(folder / 'per-query.npy', table(rows))
            (folder / 'report.json').write_text(json.dumps({'order': order, 'seed': seed, 'backend': backend}))


def test_aggregate_separates_the_query_from_the_order(tmp_path):
    write_builds(tmp_path / 'big', 'epzero')
    write_builds(tmp_path / 'small', 'epzero')
    (tmp_path / 'big' / 'sample').mkdir()
    (tmp_path / 'big' / 'sample' / 'sample.json').write_text(json.dumps({'count': 1234}))
    report = strata.analyze({'big': tmp_path / 'big', 'small': tmp_path / 'small'}, tmp_path / 'out.json', ef=None,
                            compare=('big', ['small']))
    block = report['collections']['big']
    assert block['documents'] == 1234 and report['collections']['small']['documents'] is None
    ratios = block['implementations']['epzero']['ratios']['cohort3_vs_cohort1']
    # Pooled: the shuffled builds look like an excess of later documents. Within a query there is none.
    assert ratios['pooled_ratio']['shuffled']['value'] > 1.25
    assert ratios['within_query_ratio']['shuffled']['value'] == pytest.approx(1.0, abs=.08)
    assert ratios['within_query_ratio']['appended']['value'] == pytest.approx(1.5, abs=.12)
    assert ratios['within_query_ratio']['reversed']['value'] == pytest.approx(1 / 1.5, abs=.08)
    low, high = ratios['within_query_ratio']['reversed']['interval_95']
    assert low < ratios['within_query_ratio']['reversed']['value'] < high < 1
    assert ratios['within_query_ratio']['as_arrived_minus_reversed']['interval_95'][0] > 0
    assert ratios['within_query_rate_difference']['shuffled']['value'] == pytest.approx(0.0, abs=.02)
    thirds = block['implementations']['epzero']['query_difficulty']['thirds_by_share_of_later_neighbours']
    assert thirds[0]['miss_rate_by_arrival_group'][0] < thirds[2]['miss_rate_by_arrival_group'][0]
    assert block['implementations']['epzero']['query_difficulty']['correlation_share_with_snapshot1_miss_rate'] > .8
    # The two collections were generated identically, so they do not differ.
    difference = report['between_collections']['implementations']['epzero']['within_query_ratio']['shuffled']['difference']
    assert difference['value'] == pytest.approx(0.0, abs=1e-12)
    assert json.loads((tmp_path / 'out.json').read_text())['strata'] == 'query x exact-rank band'
    with pytest.raises(ValueError, match='no build'):
        strata.tables(tmp_path / 'empty')


def test_heterogeneity_finds_a_common_ratio_and_a_band_that_differs(tmp_path):
    write_builds(tmp_path / 'common', 'epzero')
    common = strata.heterogeneity(tmp_path / 'common' / 'epzero', ef=None, draws=200)
    # The generator gives every query and band the same order effect: subgroups agree.
    appended = common['appended']
    assert all(abs(appended[f'band {lo}-{hi}']['value'] - 1.5) < .2 for lo, hi in BANDS)
    assert appended['spread across bands']['value'] < .3
    assert set(appended) >= {'third with fewest later neighbours', 'middle third', 'third with most later neighbours',
                             'spread across query thirds'}
