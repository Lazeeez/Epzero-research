"""Hand-checked cases for the post-freeze pool ledger, label agreement and insertion-pilot ratios."""
import json
import zipfile

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


def test_human_agreement_and_provenance(tmp_path):
    human = {'q1': {'a': 0, 'b': 2, 'c': 0}, 'q2': {'a': 1}}
    llm = {1: {'q1': {'a': 2, 'b': 2}}, 2: {}, 3: {'q1': {'c': 0}, 'q2': {'a': 1}}}
    qrels = {'human': {3: human}, 'gpt-oss-120b': llm}
    exact = {3: {'q1': [('b', 1.0), ('a', .5)], 'q2': [('a', 1.0)]}}
    r = heldout_pool.human_agreement(qrels, exact, ['q1', 'q2'], families=('gpt-oss-120b',), draws=20)
    assert (r['queries'], r['pairs']) == (2, 4)
    item = r['families']['gpt-oss-120b']
    # The LLM grade of a pair is taken from whichever snapshot graded it.
    assert item['overlapping_pairs'] == 4
    assert item['confusion_human_rows_llm_columns'] == [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
    assert item['binary_agreement'] == pytest.approx(.75)
    assert item['human_zero_graded_positive_by_llm'] == pytest.approx(.5)
    assert item['human_positive_graded_zero_by_llm'] == 0.0
    assert (item['human_positive_share'], item['llm_positive_share']) == (pytest.approx(.5), pytest.approx(.75))
    # Human labels: q1 ranks its only relevant document first, q2 too. Snapshot-3 LLM labels: q1 has no positive.
    assert item['exact_search_ndcg_at_10']['human'] == pytest.approx(1.0)
    assert item['exact_search_ndcg_at_10']['llm'] == pytest.approx(.5)

    (tmp_path / 'task1_longeval_adhoc-qrels-snapshot-09-11-dctr.txt').write_text('q1 0 a 0\nq9 0 z 1\n')
    (tmp_path / 'task1_longeval_adhoc-qrels-snapshot-09-11-raw.txt').write_text('q9 0 z 1\n')
    with zipfile.ZipFile(tmp_path / 'llm_synthetic_qrels.zip', 'w') as archive:
        archive.writestr('llm_synthetic_qrels/copy.qrels.txt', 'q1 0 a 0\nq1 0 b 2\nq1 0 c 0\nq2 0 a 1\nq3 0 x 1\n')
        archive.writestr('llm_synthetic_qrels/other.qrels.txt', 'q1 0 a 2\nq1 0 b 2\n')
    prov = heldout_pool.human_label_provenance(tmp_path, human)
    assert prov['human_pairs'] == 4 and prov['llm_label_files'] == 2
    assert prov['llm_files_grading_every_human_pair'] == 1 and prov['rows_in_those_files'] == [5]
    assert prov['best_exact_agreement_with_an_llm_file'] == {'file': 'copy.qrels.txt', 'agreement': 1.0}
    assert prov['click_labels']['dctr'] == {'rows': 2, 'shared_pairs': 1, 'exact_agreement': 1.0}
    assert prov['click_labels']['raw'] == {'rows': 1, 'shared_pairs': 0, 'exact_agreement': None}


def small_sample(folder, n=600, dim=16, queries=12):
    rng = np.random.default_rng(5)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    q = rng.standard_normal((queries, dim)).astype(np.float32)
    cohort = np.repeat(np.array([1, 2, 3], dtype=np.int8), n // 3)
    folder.mkdir()
    np.save(folder / 'vectors.npy', vectors)
    np.save(folder / 'cohort.npy', cohort)
    np.save(folder / 'lexical-rank.npy', rng.permutation(n).astype(np.int64))
    np.save(folder / 'queries.npy', q)
    np.save(folder / 'exact-top100.npy', np.argsort(-(q.astype(np.float64) @ vectors.astype(np.float64).T), axis=1,
                                                    kind='stable')[:, :100].astype(np.int32))
    (folder / 'sample.json').write_text(json.dumps({'count': n}))
    return folder


def test_pilot_width_sweep_and_robustness_summary(tmp_path):
    pytest.importorskip('faiss')
    sample = small_sample(tmp_path / 'sample')
    root = tmp_path / 'm8'
    for order in ('appended', 'reversed', 'shuffled'):
        report = insertion_pilot.run(sample, root / 'faiss_hnsw' / f'{order}-11', order=order, seed=11, ef=40, M=8,
                                     ef_construction=40, efs=(10, 40, 600))
        assert list(report['widths']) == ['10', '40', '600'] and report['M'] == 8
        folder = root / 'faiss_hnsw' / f'{order}-11'
        primary = np.load(folder / 'per-query.npy')
        assert (primary == np.load(folder / 'per-query-ef40.npy')).all()
        assert primary[..., 0].sum() == 12 * 100
        recalls = [report['widths'][w]['recall_at_100'] for w in ('10', '40', '600')]
        assert recalls[0] <= recalls[1] <= recalls[2] and recalls[2] > .95      # ef 600 visits every reachable document
        assert report['widths']['40']['recall_at_100'] == report['recall_at_100']
    # Same order and seed: the same graph, so the same misses.
    again = insertion_pilot.run(sample, tmp_path / 'again', order='reversed', seed=11, ef=40, M=8, ef_construction=40)
    assert (np.load(tmp_path / 'again/per-query.npy') == np.load(root / 'faiss_hnsw/reversed-11/per-query.npy')).all()
    assert 'widths' not in again
    out = insertion_pilot.robustness(sample, [root], tmp_path / 'robustness.json')
    block = out['settings']['m8']
    assert (block['M'], block['ef_construction']) == (8, 40)
    widths = block['backends']['faiss_hnsw']['widths']
    assert list(widths) == ['10', '40', '600']
    assert set(widths['10']['contrasts']) == {'cohort3_vs_cohort1', 'cohort2_vs_cohort1', 'later_vs_cohort1'}
    assert set(widths['10']['missed_by_cohort']) == {'appended-11', 'reversed-11', 'shuffled-11'}
    assert json.loads((tmp_path / 'robustness.json').read_text())['settings']['m8']['backends']['faiss_hnsw']['builds'] == 3
