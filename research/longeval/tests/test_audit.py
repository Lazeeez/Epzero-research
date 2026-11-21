"""The human-audit kit: roles, sampling, blinded pages and the analysis of returned grades."""
import json

import pytest

from research.longeval import audit


def runs():
    exact = {1: {'q1': [('a', 3.0), ('b', 2.0)], 'q2': [('x', 3.0), ('y', 2.0)]},
             2: {'q1': [('a', 3.0), ('c', 2.5)], 'q2': [('x', 3.0), ('y', 2.0)]}}
    primary = {('epzero', 'incremental', 11, 1, 'matched_recall'): {'q1': [('a', 3.0), ('c', 1.0)], 'q2': [('x', 3.0), ('z', 1.0)]},
               ('epzero', 'incremental', 11, 2, 'matched_recall'): {'q1': [('a', 3.0), ('b', 1.0)], 'q2': [('x', 3.0), ('y', 2.0)]},
               ('epzero', 'incremental', 11, 1, 'frozen'): {'q1': [('m', 3.0), ('n', 1.0)], 'q2': [('m', 3.0), ('n', 1.0)]}}
    return exact, primary


def test_roles_count_occurrences_per_pair():
    exact, primary = runs()
    arms = {'epzero_hnsw': ('epzero', 'incremental', 'matched_recall')}
    found = audit.roles(primary, exact, arms, ['q1', 'q2'])
    # Snapshot 1 drops b for c and y for z; snapshot 2 drops c for b. The fixed-width arm is not a primary arm.
    assert found == {('q1', 'b'): {'dropped': 1, 'replacement': 1}, ('q1', 'c'): {'dropped': 1, 'replacement': 1},
                     ('q2', 'y'): {'dropped': 1, 'replacement': 0}, ('q2', 'z'): {'dropped': 0, 'replacement': 1}}
    chosen, counts = audit.select(found, per_role=1, overlap=1)
    # Pairs that play both roles are left out.
    assert [(item['query_id'], item['doc_id'], item['role']) for item in chosen] == [('q2', 'y', 'dropped'), ('q2', 'z', 'replacement')]
    assert counts == {'dropped': 1, 'replacement': 1, 'both_roles_excluded': 2}
    assert all(item['selection_probability'] == 1.0 and item['overlap'] for item in chosen)
    with pytest.raises(ValueError, match='only 1 pairs'):
        audit.select(found, per_role=2)


def kit(folder, pairs=12):
    chosen = [{'query_id': f'q{n % 4}', 'doc_id': f'd{n}', 'role': audit.ROLES[n % 2], 'occurrences': 1,
               'selection_probability': .5, 'overlap': n < 4} for n in range(pairs)]
    queries = {f'q{n}': f'query {n}' for n in range(4)}
    texts = {f'd{n}': f'Title {n} </script> abstract {n}' for n in range(pairs)}
    official = {(f'q{n % 4}', f'd{n}'): n % 3 for n in range(0, pairs, 2)}          # only dropped pairs are graded
    return audit.write_kit(folder, chosen, queries, texts, official, {'arms': ['epzero_hnsw']})


def test_pages_are_blind_and_the_key_is_separate(tmp_path):
    contract = kit(tmp_path / 'kit')
    assert contract['pairs'] == 12 and contract['overlap_pairs'] == 4
    page = (tmp_path / 'kit/annotate.html').read_text(encoding='utf-8')
    data = json.loads(page.split('<script id="data" type="application/json">')[1].split('</script>')[0].replace('<\\/', '</'))
    assert [p['pair'] for p in data['pairs']] == list(range(1, 13)) and set(data['pairs'][0]) == {'pair', 'query', 'text'}
    # Nothing on the page says which pair is which, and a closing tag in a document cannot break the page.
    assert 'dropped' not in page and 'replacement' not in page and 'official' not in page
    assert any('</script>' in p['text'] for p in data['pairs'])
    second = (tmp_path / 'kit/annotate-overlap.html').read_text(encoding='utf-8')
    assert len(json.loads(second.split('type="application/json">')[1].split('</script>')[0].replace('<\\/', '</'))['pairs']) == 4
    key = [json.loads(line) for line in (tmp_path / 'kit/key.jsonl').read_text().splitlines()]
    assert sorted(row['role'] for row in key).count('dropped') == 6 and {row['pair'] for row in key} == set(range(1, 13))


def test_analysis_of_returned_grades(tmp_path):
    contract = kit(tmp_path / 'kit')
    key = {row['pair']: row for row in map(json.loads, (tmp_path / 'kit/key.jsonl').read_text().splitlines())}
    # Dropped documents graded 2, replacements graded 0, except one replacement graded 1.
    grades = {str(pair): 2 if row['role'] == 'dropped' else 0 for pair, row in key.items()}
    odd = next(pair for pair, row in key.items() if row['role'] == 'replacement')
    grades[str(odd)] = 1
    (tmp_path / 'first.json').write_text(json.dumps({'pairs_sha256': contract['pairs_sha256'], 'annotator': 'A', 'grades': grades}))
    report = audit.analyze(tmp_path / 'kit', tmp_path / 'first.json', draws=200)
    assert report['complete'] and report['roles']['dropped']['mean_grade'] == 2.0
    assert report['roles']['replacement']['mean_grade'] == pytest.approx(1 / 6)
    assert report['roles']['dropped']['officially_graded'] == 6 and report['roles']['replacement']['officially_graded'] == 0
    assert report['dropped_minus_replacement']['mean_grade_difference'] == pytest.approx(2 - 1 / 6)
    assert report['against_official_grades']['pairs'] == 6
    overlap = {str(pair): grades[str(pair)] for pair, row in key.items() if row['overlap']}
    changed = next(iter(overlap))
    overlap[changed] = (overlap[changed] + 1) % 3
    (tmp_path / 'second.json').write_text(json.dumps({'pairs_sha256': contract['pairs_sha256'], 'annotator': 'B', 'grades': overlap}))
    both = audit.analyze(tmp_path / 'kit', tmp_path / 'first.json', second=tmp_path / 'second.json', draws=50)
    assert both['between_annotators']['pairs'] == 4 and both['between_annotators']['exact_agreement'] == .75
    (tmp_path / 'other.json').write_text(json.dumps({'pairs_sha256': 'x', 'grades': grades}))
    with pytest.raises(ValueError, match='another set of pairs'):
        audit.analyze(tmp_path / 'kit', tmp_path / 'other.json')
