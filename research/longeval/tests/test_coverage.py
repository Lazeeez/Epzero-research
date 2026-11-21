"""Hand-checked cases for the validation sample, the validation measures, the coverage estimate and the grading rules."""
import json
import math
import threading

import pytest

from research.longeval import coverage, judge


def test_validation_sample_strata_and_probabilities():
    pairs = {}
    for grade in (0, 1, 2):
        for step in (1, 2, 3):
            for n in range(20):
                pairs[(f'q{n % 5}', f'd{grade}{step}{n}')] = (grade, step)
    pairs[('q0', 'top1')] = (3, 1)
    pairs[('q1', 'top2')] = (3, 2)
    excluded = {('q0', 'd0110')}
    rows = coverage.validation_sample(pairs, excluded, size=47)
    assert len(rows) == 47 and len({(row['query_id'], row['doc_id']) for row in rows}) == 47
    # Both grade-3 pairs are certain; the other 45 are five per stratum; an excluded pair is never drawn.
    top = [row for row in rows if row['official_grade'] == 3]
    assert len(top) == 2 and all(row['selection_probability'] == 1 for row in top)
    counts = {}
    for row in rows:
        counts[(row['official_grade'], row['first_graded_snapshot'])] = counts.get((row['official_grade'], row['first_graded_snapshot']), 0) + 1
    assert all(counts[(g, s)] == 5 for g in (0, 1, 2) for s in (1, 2, 3))
    assert ('q0', 'd0110') not in {(row['query_id'], row['doc_id']) for row in rows}
    first = next(row for row in rows if (row['official_grade'], row['first_graded_snapshot']) == (0, 1))
    assert first['stratum_pairs'] == 19 and first['selection_probability'] == pytest.approx(5 / 19)
    assert rows == coverage.validation_sample(pairs, excluded, size=47)
    with pytest.raises(ValueError, match='only 20 eligible'):
        coverage.validation_sample(pairs, set(), size=2 + 9 * 21)


def test_weighted_validation_measures():
    # Two strata: grade 0 sampled 1 in 10, grade 2 sampled 1 in 2. The judge is right on grade 2 and on half of grade 0.
    key, pairs, grades = [], {}, {}
    for n in range(8):
        grade = 0 if n < 4 else 2
        key.append({'pair': n, 'validation': True, 'official_grade': grade, 'selection_probability': .1 if grade == 0 else .5})
        pairs[n] = {'query_id': f'q{n}', 'doc_id': f'd{n}'}
        grades[(f'q{n}', f'd{n}')] = grade if n >= 2 else 2
    key.append({'pair': 99, 'validation': False, 'official_grade': None, 'selection_probability': None})
    pairs[99] = {'query_id': 'q0', 'doc_id': 'pool-only'}
    report = coverage.validation(key, pairs, grades, draws=200)
    # Weights 10 for grade 0 and 2 for grade 2: agreement = (2 * 10 + 4 * 2) / (4 * 10 + 4 * 2).
    assert report['validation_pairs'] == 8 and report['graded'] == 8 and report['complete'] is True
    assert report['binary_agreement'] == pytest.approx(28 / 48)
    assert report['agreement_by_official_grade'] == {'0': .5, '2': 1.0}
    assert report['unweighted_confusion_official_rows_local_columns'][0][:3] == [2, 0, 2]
    assert report['binary_agreement_threshold'] == {'required': .9, 'met': False, 'interval_includes_threshold': False}
    perfect = {(pairs[item['pair']]['query_id'], pairs[item['pair']]['doc_id']): item['official_grade'] for item in key if item['validation']}
    report = coverage.validation(key, pairs, perfect, draws=200)
    assert report['binary_agreement'] == 1.0 and report['linear_weighted_kappa'] == pytest.approx(1.0)
    assert report['linear_weighted_kappa_threshold']['met'] is True
    del perfect[('q7', 'd7')]
    assert coverage.validation(key, pairs, perfect, draws=50)['complete'] is False


def test_list_agreement_uses_the_same_ten_documents():
    exact = {s: {'q1': [('a', 3.0), ('b', 2.0)], 'q2': [('a', 3.0), ('c', 2.0)]} for s in (1, 2, 3)}
    official = {1: {'q1': {'a': 2, 'b': 1}, 'q2': {'a': 1}}, 2: {'q1': {'a': 2, 'b': 1}}, 3: {}}
    grades = {('q1', 'a'): 1, ('q1', 'b'): 2, ('q2', 'a'): 1}
    report = coverage.list_agreement(exact, official, grades, ['q1', 'q2'])
    # Only q1 is complete at snapshots 1 and 2. Official order is ideal; the local grades reverse it.
    local = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert report['by_snapshot']['1']['complete_lists'] == 1 and report['by_snapshot']['3'] == {'complete_lists': 0}
    assert report['by_snapshot']['1']['ndcg_official'] == 1.0
    assert report['by_snapshot']['1']['ndcg_local'] == pytest.approx(local)
    assert report['by_snapshot']['1']['difference'] == pytest.approx(local - 1) and report['met'] is False


def study():
    queries = ['q1', 'q2', 'q3']
    exact = {s: {q: [('a', 3.0), ('b', 2.0)] for q in queries} for s in (1, 2, 3)}
    arms = {'arm': ('epzero', 'incremental', 'matched_recall')}
    primary = {('epzero', 'incremental', seed, s, 'matched_recall'): {q: [('a', 3.0), ('c', 1.0)] for q in queries}
               for seed in (11, 29, 47) for s in (1, 2, 3)}
    official = {s: {q: {'a': 2, 'b': 1} for q in queries} for s in (1, 2, 3)}
    grades = {(q, d): g for q in queries for d, g in (('a', 2), ('b', 1), ('c', 1))}
    universe = {s: {q: ['a', 'b', 'c'] for q in queries} for s in (1, 2, 3)}
    return queries, exact, arms, primary, official, grades, universe


def test_coverage_effect_on_a_hand_checked_case():
    queries, exact, arms, primary, official, grades, universe = study()
    labels = coverage.label_sets(official, grades, universe)
    assert labels['local_full'][1]['q1'] == {'a': 2, 'b': 1, 'c': 1} and labels['local_masked'][1]['q1'] == {'a': 2, 'b': 1}
    assert labels['official_plus_filled'][1]['q1'] == {'a': 2, 'b': 1, 'c': 1} and labels['official'][1]['q1'] == {'a': 2, 'b': 1}
    report = coverage.coverage(primary, exact, official, grades, arms, queries, universe)
    arm = report['arms']['arm']
    # The run swaps b (grade 1) for c, which the release never graded but the local judge grades 1.
    ideal = 2 + 1 / math.log2(3)
    missing = 2 / ideal - 1                              # official and masked: c counts as 0
    for estimate in ('official', 'local_masked', 'condensed'):
        assert arm['gap_by_snapshot'][estimate] == pytest.approx([missing] * 3)
    for estimate in ('local_full', 'official_plus_filled'):
        assert arm['gap_by_snapshot'][estimate] == pytest.approx([0.0] * 3)
    assert arm['coverage_effect']['1']['mean'] == pytest.approx(-missing)
    assert arm['coverage_effect']['s3_minus_s1']['mean'] == pytest.approx(0.0)
    assert arm['change_s3_minus_s1']['official']['mean'] == pytest.approx(0.0)
    assert report['complete'] is True and arm['queries_without_a_positive_grade']['local_full'] == [0, 0, 0]


def test_the_scoring_universe_changes_the_result():
    # A relevant document that only another run returned belongs to a wider universe and raises the ideal DCG.
    queries, exact, arms, primary, official, grades, universe = study()
    wider = {s: {q: ['a', 'b', 'c', 'z'] for q in queries} for s in (1, 2, 3)}
    grades.update({(q, 'z'): 2 for q in queries})
    narrow = coverage.coverage(primary, exact, official, grades, arms, queries, universe)['arms']['arm']
    wide = coverage.coverage(primary, exact, official, grades, arms, queries, wider)['arms']['arm']
    assert narrow['gap_by_snapshot']['local_full'] == pytest.approx([0.0] * 3)
    # Both runs lose against the larger ideal, by the same amount: the gap stays 0, the masked gap shrinks.
    assert wide['gap_by_snapshot']['local_full'] == pytest.approx([0.0] * 3)
    assert wide['gap_by_snapshot']['local_masked'] == narrow['gap_by_snapshot']['local_masked']
    assert coverage.label_sets(official, grades, wider)['local_full'][1]['q1']['z'] == 2


def test_a_missing_grade_blocks_the_primary_report():
    queries, exact, arms, primary, official, grades, universe = study()
    del grades[('q1', 'c')]
    effect = coverage.coverage(primary, exact, official, grades, arms, queries, universe)
    # c is missing for q1 at every snapshot, and the release never graded it.
    assert effect['complete'] is False
    assert effect['missing_grades']['2'] == {'occurrences': 9, 'missing': 1, 'missing_officially_graded': 0,
                                             'missing_officially_ungraded': 1}
    checked = {'complete': True, 'validation_pairs': 10, 'graded': 10}
    with pytest.raises(ValueError, match='grades are missing'):
        coverage.require_complete(checked, effect)
    assert coverage.require_complete(checked, effect, allow_partial=True) is False
    grades[('q1', 'c')] = 1
    effect = coverage.coverage(primary, exact, official, grades, arms, queries, universe)
    assert coverage.require_complete(checked, effect) is True
    with pytest.raises(ValueError, match='1 validation pairs'):
        coverage.require_complete({'complete': False, 'validation_pairs': 10, 'graded': 9}, effect)


def grading_inputs(folder, order=range(1, 7)):
    folder.mkdir()
    rows = [{'pair': n, 'query_id': f'q{n % 2}', 'doc_id': f'd{n}', 'query': f'query {n % 2}', 'text': f'Title {n}.'} for n in order]
    judge._write_jsonl(folder / 'pairs.jsonl', rows)
    judge._write_jsonl(folder / 'key.jsonl', [{'pair': n} for n in order])
    (folder / 'universe.json').write_text(json.dumps({'pool_sets': ['reported'], 'snapshots': {}}))
    files = {name: coverage.checksum(folder / name) for name in coverage.INPUT_FILES}
    (folder / 'inputs.json').write_text(json.dumps({'files': files}))
    return folder


@pytest.fixture
def endpoint():
    server = judge.mock_server()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{server.server_address[1]}/v1'
    server.shutdown()


def test_grading_resumes_only_under_the_same_configuration(tmp_path, endpoint):
    inputs = grading_inputs(tmp_path / 'inputs')
    record = coverage.grade(inputs, tmp_path / 'out', base_url=endpoint, params={'temperature': 0}, workers=2)
    assert record['rounds'] == [{'attempt': 1, 'requested': 6}] and record['pairs_without_a_usable_answer'] == 0
    bound = json.loads((tmp_path / 'out/binding.json').read_text())
    assert bound['request_parameters'] == {'temperature': 0} and bound['model_revision'] == judge.MODEL_REVISION
    # An interrupted run: drop the last two answers and start again. Only those two are asked.
    path = tmp_path / 'out/responses.jsonl'
    path.write_text(''.join(line + '\n' for line in path.read_text().splitlines()[:4]))
    again = coverage.grade(inputs, tmp_path / 'out', base_url=endpoint, params={'temperature': 0}, workers=2)
    assert again['rounds'] == [{'attempt': 1, 'requested': 2}] and again['pairs_with_a_usable_answer'] == 6
    grades, answers = coverage.local_grades(inputs, tmp_path / 'out')
    assert len(grades) == 6 and answers['pairs_without_a_usable_answer'] == 0 and answers['surplus_usable_answers'] == 0
    # Another decoding setting, or the same pairs in another order, must not reuse this folder.
    with pytest.raises(ValueError, match='bound to another grading configuration'):
        coverage.grade(inputs, tmp_path / 'out', base_url=endpoint, params={}, workers=2)
    reordered = grading_inputs(tmp_path / 'reordered', order=[2, 1, 3, 4, 5, 6])
    with pytest.raises(ValueError, match='bound to another grading configuration'):
        coverage.grade(reordered, tmp_path / 'out', base_url=endpoint, params={'temperature': 0}, workers=2)
    (tmp_path / 'orphan').mkdir()
    (tmp_path / 'orphan/responses.jsonl').write_text(path.read_text())
    with pytest.raises(ValueError, match='without a binding record'):
        coverage.grade(inputs, tmp_path / 'orphan', base_url=endpoint, params={'temperature': 0})
    (inputs / 'pairs.jsonl').write_text((inputs / 'pairs.jsonl').read_text().replace('Title 3', 'Title three'))
    with pytest.raises(ValueError, match='input checksum mismatch'):
        coverage.grade(inputs, tmp_path / 'new', base_url=endpoint, params={})


def test_unparsed_answers_are_asked_again_and_never_become_a_grade(tmp_path, monkeypatch):
    inputs = grading_inputs(tmp_path / 'inputs')
    calls = {}

    def ask(base_url, model, row, params):
        calls[row['pair']] = calls.get(row['pair'], 0) + 1
        # Pair 1 never gives a category, pair 2 does on its second attempt, the others at once.
        text = 'No category.' if row['pair'] == 1 or (row['pair'] == 2 and calls[2] == 1) else 'Reasoning.\n2'
        return {'pair': row['pair'], 'status': 'ok', 'message': {'content': text}, 'finish_reason': 'stop'}

    monkeypatch.setattr(coverage.judge, 'ask', ask)
    record = coverage.grade(inputs, tmp_path / 'out', base_url='unused', params={}, workers=1)
    assert record['rounds'] == [{'attempt': 1, 'requested': 6}, {'attempt': 2, 'requested': 2}, {'attempt': 3, 'requested': 1}]
    assert calls == {1: 3, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1} and record['pairs_without_a_usable_answer'] == 1
    grades, answers = coverage.local_grades(inputs, tmp_path / 'out')
    assert ('q1', 'd1') not in grades and grades[('q0', 'd2')] == 2 and answers['responses'] == 9
    # A further start does not ask the exhausted pair a fourth time.
    assert coverage.grade(inputs, tmp_path / 'out', base_url='unused', params={}, workers=1)['rounds'] == []
    assert calls[1] == 3
