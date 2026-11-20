import hashlib
import json

import numpy as np
import pytest

from research.longeval import pseudo_queries as pq
from research.longeval.query_diagnostic import PROMPT


def test_forms_collapse_whitespace_and_drop_stopwords():
    assert pq.forms('  The  Effect of Climate\nChange on  Coral Reefs ') == {
        'title': 'The Effect of Climate Change on Coral Reefs',
        'keywords4': 'effect climate change coral'}
    assert pq.keywords("Children's self-reported well-being: a case study") == "children's self-reported well-being"
    assert pq.keywords('COVID-19 vaccine uptake (2021) in rural India') == 'covid-19 vaccine uptake 2021'
    assert pq.forms('   ') == {'title': None, 'keywords4': None}
    assert pq.keywords('The of and') is None
    assert pq.keywords('Mark Twain', count=4) == 'mark twain'


def test_hash_order_is_deterministic_and_salted():
    ids = [str(i) for i in range(50)]
    assert pq.hash_order(ids) == pq.hash_order(list(reversed(ids)))
    expected = sorted(ids, key=lambda k: hashlib.sha256((pq.SALT + k).encode()).digest())
    assert pq.hash_order(ids) == expected != sorted(ids)


def test_build_records_counts_every_exclusion_and_prompts_text():
    titles = {'d1': 'Deep Learning for Protein Folding', 'd2': '   ', 'd3': 'mark twain', 'd4': ' '.join(['word'] * 70)}
    real = {pq.normalized('Mark Twain')}
    records, excluded = pq.build_records({'A': (['d1', 'd2', 'd3', 'd4', 'd5'], ('title', 'keywords4'))}, titles, real)
    ids = [r['id'] for r in records['A']]
    assert ids == ['A:title:d1', 'A:keywords4:d1', 'A:keywords4:d4']
    assert excluded['A'] == {'missing_title': 1, 'empty_form': 2, 'too_long': 1, 'real_query_collision': 2}
    for row in records['A']:
        assert row['text'].startswith(PROMPT)
        assert row['text_sha256'] == hashlib.sha256(row['text'].encode()).hexdigest()


def test_length_batches_partition_equal_lengths():
    lengths = [3, 5, 3, 3, 5, 7, 3]
    groups = pq.length_batches(lengths, 2)
    flat = sorted(i for g in groups for i in g)
    assert flat == list(range(len(lengths)))
    assert all(len(g) <= 2 and len({lengths[i] for i in g}) == 1 for g in groups)
    assert groups == [[0, 2], [3, 6], [1, 4], [5]]
    with pytest.raises(ValueError):
        pq.length_batches([3, 0], 2)
    with pytest.raises(ValueError):
        pq.length_batches([3], 0)


def fake_tokenize(texts):
    return [[ord(c) % 97 + 1 for c in text] for text in texts]


def fake_forward(rows, dim=8):
    assert len({len(r) for r in rows}) == 1, 'batches must not mix lengths'
    out = np.array([[sum(r) + j * len(r) + 1 for j in range(dim)] for r in rows], dtype=np.float64)
    return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)


def test_encode_rows_restores_order_and_matches_unbatched():
    texts = ['ab', 'abcd', 'cd', 'x', 'wxyz', 'q']
    batched, lengths = pq.encode_rows(texts, fake_tokenize, fake_forward, 2, dim=8)
    single = np.vstack([fake_forward([fake_tokenize([t])[0]]) for t in texts])
    assert lengths == [2, 4, 2, 1, 4, 1]
    assert np.array_equal(batched, single)


def test_encode_rows_rejects_bad_batches():
    with pytest.raises(ValueError):
        pq.encode_rows(['ab'], fake_tokenize, lambda rows: np.ones((len(rows), 8), np.float32), 2, dim=8)
    with pytest.raises(ValueError):
        pq.encode_rows(['ab'], fake_tokenize, lambda rows: np.full((len(rows), 8), np.nan, np.float32), 2, dim=8)
    with pytest.raises(ValueError):
        pq.encode_rows(['ab'], fake_tokenize, lambda rows: fake_forward(rows)[:, :4], 2, dim=8)


def test_cosine_gate():
    a = np.random.default_rng(1).normal(size=(5, 16)).astype(np.float32)
    assert pq.cosine_check(a, a.copy())['passed']
    b = a.copy()
    b[0, 0] += 0.5
    assert not pq.cosine_check(a, b)['passed']


def write_inputs(folder):
    titles = {'d1': 'Graph Search for Vectors', 'd2': 'Coral Reef Ecology'}
    records, excluded = pq.build_records({'A': (['d1', 'd2'], ('title', 'keywords4')), 'B': (['d2'], ('keywords4',))}, titles, set())
    folder.mkdir()
    for name, rows in records.items():
        pq.write_jsonl(folder / f'records-{name}.jsonl', rows)
    pq.write_jsonl(folder / 'reference-records.jsonl', [{'id': 'q', 'text': PROMPT + 'x'}])
    np.save(folder / 'reference-vectors.npy', np.ones((1, 4), np.float32))
    files = ('records-A.jsonl', 'records-B.jsonl', 'reference-records.jsonl', 'reference-vectors.npy')
    (folder / 'pseudo-inputs.json').write_text(json.dumps({'status': 'prepared', 'prompt': PROMPT,
        'counts': {n: len(r) for n, r in records.items()}, 'excluded': excluded,
        'files': {n: pq.checksum(folder / n) for n in files}}))


def test_validate_inputs_detects_tampering(tmp_path):
    folder = tmp_path / 'inputs'
    write_inputs(folder)
    contract, sets = pq.validate_inputs(folder)
    assert contract['counts'] == {'A': 4, 'B': 1} and len(sets['A']) == 4
    path = folder / 'records-B.jsonl'
    path.write_text(path.read_text().replace('coral', 'koral'))
    with pytest.raises(ValueError, match='checksum'):
        pq.validate_inputs(folder)
