import hashlib
import json

import numpy as np
import pytest

from research.longeval import test_queries as tq
from research.longeval.query_diagnostic import PROMPT


def test_records_are_sorted_and_prompted():
    rows = tq.records({'b2': 'mark twain', 'a1': 'coral reefs'})
    assert [r['id'] for r in rows] == ['a1', 'b2']
    assert rows[0]['text'] == PROMPT + 'coral reefs'
    assert rows[0]['text_sha256'] == hashlib.sha256(rows[0]['text'].encode()).hexdigest()


def test_cosine_gate():
    a = np.random.default_rng(3).normal(size=(4, 8)).astype(np.float32)
    assert tq.cosine_check(a, a * 2)['passed']          # scale-invariant
    b = a.copy()
    b[1] = -b[1]
    assert not tq.cosine_check(a, b)['passed']


def make_inputs(folder):
    folder.mkdir()
    tq.write_jsonl(folder / 'test-records.jsonl', tq.records({'q1': 'a', 'q2': 'b'}))
    tq.write_jsonl(folder / 'training-records.jsonl', tq.records({'t1': 'c'}))
    np.save(folder / 'training-vectors.npy', np.ones((1, 4), np.float32))
    (folder / 'training-ids.json').write_text(json.dumps(['t1']))
    names = ('test-records.jsonl', 'training-records.jsonl', 'training-vectors.npy', 'training-ids.json')
    (folder / 'test-inputs.json').write_text(json.dumps({'status': 'prepared', 'prompt': PROMPT, 'test_count': 2,
        'files': {n: tq.checksum(folder / n) for n in names}}))


def test_validate_inputs_rejects_tampering(tmp_path):
    folder = tmp_path / 'inputs'
    make_inputs(folder)
    assert tq.validate_inputs(folder)['test_count'] == 2
    (folder / 'training-ids.json').write_text(json.dumps(['t2']))
    with pytest.raises(ValueError, match='checksum'):
        tq.validate_inputs(folder)
