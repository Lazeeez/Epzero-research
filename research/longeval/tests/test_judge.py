"""The judge's prompt and parser are pinned to the organisers' code; the rest is exercised against a stand-in endpoint."""
import hashlib
import json
import threading

import pytest

from research.longeval import judge


def test_prompt_is_the_organisers_prompt():
    # SHA-256 of the organisers' `umbrella_zeroshot_no_desc_no_narrative` rendered with these two
    # placeholders (longeval-code, clef26/topic-extraction/evaluation/prompts.py, commit b5f11fd2d58e).
    rendered = judge.prompt('{query}', '{document}')
    assert hashlib.sha256(rendered.encode()).hexdigest() == '7407ffe0bc62aeb8187055905073d14cb957ae0b6b0e6a7c86bd5a5841839b34'
    assert len(rendered) == 957 and rendered.endswith('Query: {query}\nDocument: {document}\n\nExplanation:\n    ')
    assert rendered.startswith('You are an expert judge of a content.')
    assert judge.prompt('a b', 'T. A.') == rendered.replace('{query}', 'a b').replace('{document}', 'T. A.')


@pytest.mark.parametrize('response, expected', [
    ('', (0, 0)),
    ('no digits here', (0, 0)),
    ('score is 5', (0, 0)),
    ('2', (2, 1)),
    ('Explanation ...\n1', (1, 1)),
    ('The passage is dedicated to the query.\n\n**2**', (2, 1)),
    ('Final score: 3', (3, 1)),
    ('category 12', (1, 1)),
    # An earlier pattern wins over the category on the last line: the parser is kept as the organisers ran it.
    ('It is 2 because\n1', (2, 1)),
    ('I think so 2 or maybe 1\n0', (2, 1)),
    ('It ranges from 0 to 2 here.\n1', (2, 1)),
])
def test_parser_follows_umbrela(response, expected):
    assert judge.parse_grade(response) == expected


def test_parser_patterns_are_pinned():
    # The 24 patterns of UMBRELA's parser, in order, as in the organisers' llm_predictions.py.
    assert len(judge._PATTERNS) == 24
    assert hashlib.sha256('\n'.join(judge._PATTERNS).encode()).hexdigest() == '52d0816068a37ce8d690d0c9c50e03640232b7f4994915f26a0894b3f6ed235c'


@pytest.mark.parametrize('response, expected', [
    ('Reasoning.\n2', 2), ('Reasoning.\n\n**1**\n', 1), ('Reasoning.\nCategory: 0', 0), ('Reasoning.\nRelevance category 2.', 2),
    ('Reasoning without a final line', None), ('It is 2 because\n1', 1), ('', None), ('4', None),
])
def test_last_line_grade(response, expected):
    assert judge.last_line_grade(response) == expected


def test_official_pairs_and_selection():
    judged = {1: {'q1': {'a': 2, 'b': 0}}, 2: {'q1': {'a': 2, 'c': 1}, 'q2': {'a': 3}}, 3: {'q2': {'d': 0}}}
    pairs = judge.official_pairs(judged)
    assert pairs == {('q1', 'a'): (2, 1), ('q1', 'b'): (0, 1), ('q1', 'c'): (1, 2), ('q2', 'a'): (3, 2), ('q2', 'd'): (0, 3)}
    chosen = judge.select_pairs(pairs, {('q1', 'b')}, per_stratum=1)
    # Grade 3 and pooled pairs are never chosen; one pair per (grade, first snapshot) stratum.
    assert sorted(chosen) == [('q1', 'a', (2, 1)), ('q1', 'c', (1, 2)), ('q2', 'd', (0, 3))]
    assert chosen == judge.select_pairs(pairs, {('q1', 'b')}, per_stratum=1)
    with pytest.raises(ValueError, match='only 1 eligible'):
        judge.select_pairs(pairs, set(), per_stratum=2)
    judged[3]['q1'] = {'a': 1}
    with pytest.raises(ValueError, match='conflicting official grades'):
        judge.official_pairs(judged)


def small_inputs(folder, pairs=12):
    rows = [{'pair': n, 'query_id': f'q{n % 4}', 'doc_id': f'd{n}', 'query': f'query {n % 4}', 'text': f'Title {n} Abstract {n}.'}
            for n in range(1, pairs + 1)]
    reference = [{'pair': n, 'official_grade': n % 3, 'first_graded_snapshot': 1 + n % 3} for n in range(1, pairs + 1)]
    return judge.write_inputs(folder, rows, reference, {'family': judge.FAMILY})


def test_inputs_are_validated(tmp_path):
    contract = small_inputs(tmp_path / 'inputs')
    assert judge.validate_inputs(tmp_path / 'inputs') == contract
    assert contract['pairs'] == 12 and contract['queries'] == 4
    # The judge's file holds no grade.
    assert all('grade' not in key for row in judge._read_jsonl(tmp_path / 'inputs/pairs.jsonl') for key in row)
    pairs = tmp_path / 'inputs/pairs.jsonl'
    pairs.write_text(pairs.read_text().replace('Abstract 3', 'Abstract three'))
    with pytest.raises(ValueError, match='input checksum mismatch'):
        judge.validate_inputs(tmp_path / 'inputs')


def test_pilot_runs_against_a_stand_in_endpoint(tmp_path):
    small_inputs(tmp_path / 'inputs')
    server = judge.mock_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_address[1]}/v1'
        record = judge.judge(tmp_path / 'inputs', tmp_path / 'out', base_url=url, workers=3)
        assert record['status'] == 'completed' and [run['requested'] for run in record['runs']] == [12] * 4
        # A second invocation finds every answer present and asks for nothing.
        again = judge.judge(tmp_path / 'inputs', tmp_path / 'out', base_url=url, workers=3)
        assert [run['requested'] for run in again['runs']] == [0] * 4 and [run['already_done'] for run in again['runs']] == [12] * 4
    finally:
        server.shutdown()
    first = judge._read_jsonl(tmp_path / 'out/responses-temperature_0-r1.jsonl')
    assert sorted(row['pair'] for row in first) == list(range(1, 13)) and all(row['status'] == 'ok' for row in first)
    report = judge.analyze(tmp_path / 'inputs', tmp_path / 'out')
    assert report['kind'].startswith('PILOT') and len(report['runs']) == 4
    for run in report['runs']:
        assert run['answered'] == 12 and run['unparsed'] == 0 and run['without_final_content'] == 0
        assert run['last_line_is_a_category'] == 12 and run['parser_differs_from_last_line'] == 0
        assert sum(run['grade_counts'].values()) == 12 and run['against_official']['pairs'] == 12
    stable = next(item for item in report['between_repeats'] if item['condition'] == 'temperature_0')
    assert stable['identical_answers'] == 1.0 and stable['exact_agreement'] == 1.0
    assert json.loads((tmp_path / 'out/report.json').read_text())['prompt_sha256'] == report['prompt_sha256']


def test_unreachable_endpoint_is_recorded_not_raised(tmp_path, monkeypatch):
    small_inputs(tmp_path / 'inputs', pairs=3)
    monkeypatch.setattr(judge.time, 'sleep', lambda seconds: None)
    record = judge.judge(tmp_path / 'inputs', tmp_path / 'out', base_url='http://127.0.0.1:9/v1', workers=2,
                         conditions={'temperature_0': {'temperature': 0}}, repeats=1)
    rows = judge._read_jsonl(tmp_path / 'out' / record['runs'][0]['file'])
    assert [row['status'] for row in rows] == ['failed'] * 3 and all(row['attempts'] == 3 for row in rows)
    report = judge.analyze(tmp_path / 'inputs', tmp_path / 'out')
    assert report['runs'][0]['failed_requests'] == 3 and report['runs'][0]['unparsed'] == 3


def test_model_manifest_is_pinned():
    from pathlib import Path
    files = judge.model_manifest(Path(judge.__file__).with_name('judge-model.json'))
    shards = [name for name in files if name.startswith('model-') and name.endswith('.safetensors')]
    assert len(shards) == 15 and 'tokenizer.json' in files and 'config.json' in files
    assert all(size > 0 and len(digest) == 64 for size, digest in files.values())
    assert 60e9 < sum(size for size, _ in files.values()) < 70e9
