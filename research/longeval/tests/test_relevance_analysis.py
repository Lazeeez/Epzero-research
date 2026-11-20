import pytest

from research.longeval.relevance_analysis import score


def test_neighbor_loss_need_not_be_relevance_loss():
    ref = {'q': [(f'd{i:03}', float(100-i)) for i in range(100)]}
    run = {'q': [('relevant', 101.)] + ref['q'][:99]}
    result = score(run, ref, {'q': {'relevant': 2, 'd000': 1}}, ['q'])
    assert result['mean']['ANN_recall@100'] == .99
    assert result['mean']['nDCG@10'] == 1.
    assert result['mean']['R@100'] == 1.
    assert score(ref, ref, {'q': {'relevant': 2, 'd000': 1}}, ['q'])['mean']['nDCG@10'] < 1.


def test_rejects_missing_queries_and_duplicate_documents():
    ref = {'q': [(str(i), float(100-i)) for i in range(100)]}
    with pytest.raises(ValueError, match='population'):
        score({}, ref, {'q': {}}, ['q'])
    with pytest.raises(ValueError, match='unique'):
        score({'q': [('x', 1.)] * 100}, ref, {'q': {}}, ['q'])
