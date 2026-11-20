"""Hand-checked cases for the post-freeze pool ledger, label agreement and insertion-pilot ratios."""
import numpy as np
import pytest

from research.longeval import heldout_pool, insertion_pilot
from research.longeval.heldout_robustness import relative_change


def small_pool():
    ids = ['q1']
    exact = {s: {'q1': [('a', 1.0), ('b', .9)]} for s in (1, 2, 3)}
    primary = {('epzero', 'incremental', 11, 1, 'frozen'): {'q1': [('a', 1.0), ('c', .5)]},
               ('faiss_ivf', 'frozen_incremental', 11, 1, 'all_lists'): {'q1': [('d', 1.0)]}}
    extension = {('sweep', 'epzero', 'incremental', 11, 2, 'ef50'): {'q1': [('e', 1.0)]}}
    judged = {1: {'q1': {'a': 2}}, 2: {'q1': {'a': 2, 'c': 1}}, 3: {'q1': {}}}
    return ids, exact, primary, extension, judged


def test_ledger_keeps_occurrences_and_pairs_apart():
    ids, exact, primary, extension, judged = small_pool()
    rows = heldout_pool.ledger(ids, exact, primary, extension)
    assert len(rows) == 9
    assert rows[('q1', 'a', 1)] == {'set': 0, 'rank': 1, 'sources': {'exact', 'epzero/incremental/frozen/seed11'}}
    assert rows[('q1', 'd', 1)]['set'] == 1 and rows[('q1', 'e', 2)]['set'] == 2
    reported = heldout_pool.counts(rows, judged, ids, 0)
    assert reported['occurrences'] == {
        'total': 7, 'judged_at_that_snapshot': 2, 'unjudged_at_that_snapshot': 5,
        'unjudged_fillable_by_carry_over': 2, 'unjudged_never_judged': 3,
        'by_snapshot': {'1': {'total': 3, 'unjudged': 2, 'fillable_by_carry_over': 1, 'never_judged': 1},
                        '2': {'total': 2, 'unjudged': 1, 'fillable_by_carry_over': 0, 'never_judged': 1},
                        '3': {'total': 2, 'unjudged': 2, 'fillable_by_carry_over': 1, 'never_judged': 1}}}
    assert reported['pairs'] == {
        'total': 3, 'judged_at_some_snapshot': 2, 'never_judged': 1, 'with_an_unjudged_occurrence': 3,
        'with_an_unjudged_occurrence_fillable_by_carry_over': 2, 'with_an_unjudged_occurrence_never_judged': 1}
    wider = heldout_pool.counts(rows, judged, ids, 1)
    assert wider['occurrences']['total'] == 8 and wider['pairs']['total'] == 4 and wider['pairs']['never_judged'] == 2
    assert heldout_pool.counts(rows, judged, ids, 2)['pairs']['total'] == 5
    assert heldout_pool.counts(rows, judged, [], 2)['occurrences']['total'] == 0


def test_carried_labels_within_the_pool_and_wherever_the_document_exists():
    ids, exact, primary, extension, judged = small_pool()
    rows = heldout_pool.ledger(ids, exact, primary, extension)
    labels, added = heldout_pool.carried_labels(rows, judged, ids, 0)
    assert labels == {1: {'q1': {'a': 2, 'c': 1}}, 2: {'q1': {'a': 2, 'c': 1}}, 3: {'q1': {'a': 2}}}
    assert added == {'1': 1, '2': 0, '3': 1}
    # Document c arrives at snapshot 2: its grade applies at snapshots 2 and 3, never at snapshot 1.
    labels, added = heldout_pool.carried_labels(rows, judged, ids, 0, first={'a': 1, 'b': 1, 'c': 2, 'd': 1, 'e': 1})
    assert labels == {1: {'q1': {'a': 2}}, 2: {'q1': {'a': 2, 'c': 1}}, 3: {'q1': {'a': 2, 'c': 1}}}
    assert added == {'1': 0, '2': 0, '3': 2}
    assert judged == small_pool()[4]                     # the official labels are not modified


def test_weighted_kappa():
    assert heldout_pool.weighted_kappa([[5, 0], [0, 5]]) == pytest.approx(1.0)
    # Observed agreement 0.7, chance agreement 0.5.
    assert heldout_pool.weighted_kappa([[20, 5], [10, 15]]) == pytest.approx(.4)
    # Three grades: a one-step disagreement keeps half credit under linear weights.
    # Observed 3.5 / 4 = .875; expected 9 / 16 = .5625.
    assert heldout_pool.weighted_kappa([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) == pytest.approx(.3125 / .4375)


def test_label_agreement_between_two_label_sets():
    qrels = {'gpt-oss-120b': {1: {'q1': {'a': 2, 'b': 0}, 'q2': {'a': 1}}, 2: {'q1': {'a': 2}}, 3: {}},
             'gpt-oss-20b': {1: {'q1': {'a': 2, 'b': 1}, 'q2': {'a': 1}}, 2: {'q1': {'a': 2}}, 3: {}}}
    exact = {s: {'q1': [('a', 1.0), ('b', .5)], 'q2': [('a', 1.0)]} for s in (1, 2, 3)}
    r = heldout_pool.label_agreement(qrels, exact, ['q1', 'q2'], draws=20)
    assert r['pairs'] == 3 and r['queries'] == 2
    assert r['exact_agreement'] == pytest.approx(2 / 3) and r['binary_agreement'] == pytest.approx(2 / 3)
    assert r['confusion_reference_rows_other_columns'] == [[0, 1, 0], [0, 1, 0], [0, 0, 1]]
    # Snapshot 1: both lists are fully graded; q1 scores 1 under either label set, q2 scores 1.
    assert r['exact_top10_complete_lists']['1'] == {
        'complete_lists': 2, 'ndcg_reference': 1.0, 'ndcg_other': 1.0, 'difference': 0.0,
        'mean_absolute_difference_per_query': 0.0}
    # Snapshot 2 grades only one of the two documents in the q1 list, and nothing for q2.
    assert r['exact_top10_complete_lists']['2'] == {'complete_lists': 0}
    qrels['gpt-oss-20b'][2]['q1']['a'] = 1
    with pytest.raises(ValueError, match='conflicting grades'):
        heldout_pool.label_agreement(qrels, exact, ['q1', 'q2'], draws=20)


def test_insertion_orders():
    cohort = np.array([1, 2, 3, 1, 2, 3], dtype=np.int8)
    lexical = np.array([5, 4, 3, 2, 1, 0])
    assert insertion_pilot.insertion_order(lexical, cohort, 'appended', 11).tolist() == [3, 0, 4, 1, 5, 2]
    # Only the arrival blocks are reversed; the lexical order inside each block is kept.
    assert insertion_pilot.insertion_order(lexical, cohort, 'reversed', 11).tolist() == [5, 2, 4, 1, 3, 0]
    shuffled = insertion_pilot.insertion_order(lexical, cohort, 'shuffled', 11)
    assert sorted(shuffled.tolist()) == list(range(6))
    assert shuffled.tolist() == insertion_pilot.insertion_order(lexical, cohort, 'shuffled', 11).tolist()
    with pytest.raises(ValueError):
        insertion_pilot.insertion_order(lexical, cohort, 'sorted', 11)


def test_rank_standardized_miss_ratio():
    total = np.array([[[100, 1], [200, 4], [300, 30]],       # snapshot-1 arrivals: miss rates .01, .02, .1
                      [[10, 0], [0, 0], [10, 3]],
                      [[50, 2], [50, 2], [100, 10]]])
    assert insertion_pilot._ratio(total, 3, 1) == pytest.approx(14 / 11.5)
    assert insertion_pilot._ratio(total, 2, 1) == pytest.approx(3 / 1.1)
    assert insertion_pilot._ratio(total, (2, 3), 1) == pytest.approx(17 / 12.6)
    # A query bootstrap of identical queries has no spread.
    per_query = np.stack([total] * 8)
    r = insertion_pilot.ratio(per_query, 3, 1, draws=50)
    assert r['ratio'] == pytest.approx(14 / 11.5) and r['interval_95'] == pytest.approx([14 / 11.5] * 2)


def test_relative_change():
    assert relative_change(.25, .20) == pytest.approx(.2)
    assert relative_change(.25, .30) == pytest.approx(-.2)
