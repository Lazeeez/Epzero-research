import copy

import pytest

from research.longeval.native_cost_verification import verify_cohort


def evidence():
    run = {'q': [(str(i), 100.-i) for i in range(100)]}
    point = {'queries': [{'query_id': 'q', 'native_call_nanoseconds': [30, 10, 20],
                         'epzero_distance_computations': 200, 'epzero_nodes_expanded': 120}],
             'ann_recall_per_query': {'q': 1.}, 'ann_recall_mean': 1.}
    return point, run, copy.deepcopy(run)


def test_recounts_recall_and_uses_per_query_median():
    point, run, reference = evidence()
    value = verify_cohort(point, run, reference, ['q'], 'epzero')
    assert value[0]['median_nanoseconds'] == 20 and value[0]['recall'] == 1


@pytest.mark.parametrize('kind', ['missing_query', 'duplicate_doc', 'wrong_recall', 'missing_repeat', 'wrong_counter', 'nonfinite'])
def test_rejects_corrupted_cohort(kind):
    point, run, reference = evidence()
    if kind == 'missing_query': point['queries'] = []
    if kind == 'duplicate_doc': run['q'][1] = run['q'][0]
    if kind == 'wrong_recall': point['ann_recall_per_query']['q'] = .99
    if kind == 'missing_repeat': point['queries'][0]['native_call_nanoseconds'].pop()
    if kind == 'wrong_counter': point['queries'][0]['faiss_native_ndis'] = 200
    if kind == 'nonfinite': run['q'][0] = (run['q'][0][0], float('nan'))
    with pytest.raises(ValueError): verify_cohort(point, run, reference, ['q'], 'epzero')
