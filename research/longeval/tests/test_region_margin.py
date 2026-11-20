import numpy as np
import pytest

from research.longeval.region_margin import edge_certificate


def test_bound_attained_in_worst_ball_direction():
    c, s, d = np.array([.2, .8]), np.array([-.3, .5]), np.array([.9, -.1])
    r = .27
    result = edge_certificate(c, s, d, r)
    worst = c - r * (d-s) / np.linalg.norm(d-s)
    assert float(worst @ (d-s)) == pytest.approx(result['lower_progress'])
    assert result['lower_progress'] <= result['center_progress']


def test_positive_bound_and_nonprogress_edge():
    assert edge_certificate([1., 0.], [0., 0.], [1., 0.], .5)['certified_positive_progress']
    assert not edge_certificate([1., 0.], [1., 0.], [0., 0.], 0.)['certified_positive_progress']
    assert not edge_certificate([1., 0.], [1., 0.], [1., 0.], 0.)['certified_positive_progress']


def test_failed_certificate_does_not_imply_failed_neighbor_query():
    c, q, s, d = np.array([1., 0.]), np.array([1., 2.]), np.zeros(2), np.array([1., 0.])
    result = edge_certificate(c, s, d, float(np.linalg.norm(q-c)))
    assert not result['certified_positive_progress']
    assert q @ (d-s) > 0


@pytest.mark.parametrize('center,source,destination,radius', [
    ([1.], [0., 0.], [1.], 0.), ([float('nan')], [0.], [1.], 0.),
    ([1.], [0.], [1.], -1.), ([1.], [0.], [1.], float('inf')),
])
def test_invalid_inputs_rejected(center, source, destination, radius):
    with pytest.raises(ValueError): edge_certificate(center, source, destination, radius)
