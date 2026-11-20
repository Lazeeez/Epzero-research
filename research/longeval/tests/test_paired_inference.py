import numpy as np
import pytest

from research.longeval.paired_inference import holm_adjust, paired_query_estimate


def test_holm_known_example_and_order():
    assert holm_adjust([.01, .04, .03]) == pytest.approx([.03, .06, .06])
    assert holm_adjust([1., 0., .5]) == [1., 0., 1.]


def test_null_effect_is_exactly_null():
    result = paired_query_estimate(['a', 'b', 'c'], [11, 29, 47], np.zeros((3, 3)),
                                   bootstrap_draws=500, randomization_draws=500)
    assert result['mean_difference'] == 0
    assert result['percentile_interval'] == [0., 0.]
    assert result['two_sided_randomization_p'] == 1.


def test_seeds_are_not_independent_query_observations():
    values = np.array([[-.4, .2], [.1, .3], [.8, .4]])
    common = dict(bootstrap_draws=1000, randomization_draws=1000)
    one = paired_query_estimate(['a', 'b', 'c'], [11, 29], values, **common)
    two = paired_query_estimate(['a', 'b', 'c'], [11, 29, 47, 83], np.tile(values, (1, 2)), **common)
    assert one['query_count'] == two['query_count'] == 3
    assert one['mean_difference'] == pytest.approx(two['mean_difference'])
    assert one['percentile_interval'] == pytest.approx(two['percentile_interval'])
    assert one['two_sided_randomization_p'] == two['two_sided_randomization_p']


def test_sign_symmetry_and_reproducibility():
    values = np.arange(1, 11, dtype=float).reshape(5, 2)
    common = dict(bootstrap_draws=1000, randomization_draws=1000)
    positive = paired_query_estimate(list('abcde'), [11, 29], values, **common)
    negative = paired_query_estimate(list('abcde'), [11, 29], -values, **common)
    assert positive == paired_query_estimate(list('abcde'), [11, 29], values, **common)
    assert positive['mean_difference'] == -negative['mean_difference']
    assert positive['two_sided_randomization_p'] == negative['two_sided_randomization_p']


def test_reject_incomplete_or_duplicate_samples():
    for ids, values in [(['a', 'a'], [[1], [2]]), (['a', 'b'], [[1], [np.nan]]), (['a'], [[1]])]:
        with pytest.raises(ValueError):
            paired_query_estimate(ids, [11], values)
    with pytest.raises(ValueError):
        holm_adjust([1.2])
