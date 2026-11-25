import gzip
import json

import numpy as np
import pytest

from research.longeval import bge_sample, insertion_pilot


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """insertion_pilot.prepare_full (frozen) on stand-in data, as test_post_freeze_analyses runs it."""
    from types import SimpleNamespace
    from research.longeval import heldout, heldout_exploratory, mapped, query_diagnostic
    rng = np.random.default_rng(4)
    n, dim = 400, 8
    doc_ids = [f'doc{i:04d}' for i in rng.permutation(n)]
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors[7] = vectors[3]                                    # a tie: broken by descending document ID
    test_q = rng.standard_normal((6, dim)).astype(np.float32)
    train_q = rng.standard_normal((4, dim)).astype(np.float32)
    test_ids, train_ids = [f't{i}' for i in range(6)], [f'r{i}' for i in range(4)]

    def reference(q):
        scores = vectors.astype(np.float64) @ q.astype(np.float64)
        order = sorted(range(n), key=lambda i: (-scores[i], [-ord(c) for c in doc_ids[i]]))[:100]
        return [(doc_ids[i], float(scores[i])) for i in order]

    class Snapshot:
        def __init__(self, *args, **kwargs):
            self.ids, self.vectors = doc_ids, vectors

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False
    refs = [None, None, {q: reference(v) for q, v in zip(test_ids, test_q)}]
    monkeypatch.setattr(mapped, 'MappedSnapshot', Snapshot)
    monkeypatch.setattr(heldout_exploratory, 'cohorts', lambda dense: {d: 1 + i % 3 for i, d in enumerate(doc_ids)})
    monkeypatch.setattr(heldout, 'load_heldout', lambda inputs: (
        {heldout.ROLE: SimpleNamespace(ids=test_ids, vectors=test_q)}, refs, None, None))
    monkeypatch.setattr(query_diagnostic, 'load_queries', lambda *a: (
        train_ids, train_q, {'dtype': 'float32', 'device': 'cuda', 'tf32': False}))
    monkeypatch.setattr(heldout, '_development_reference',
                        lambda inputs, step: ({}, {q: reference(v) for q, v in zip(train_ids, train_q)}))
    insertion_pilot.prepare_full(tmp_path / 'dense', tmp_path / 'inputs', tmp_path / 'qwen')
    # The same data in the BGE encoding job's layout; its query files are in another (sorted, reversed) order.
    bge = tmp_path / 'bge'
    bge.mkdir()
    np.save(bge / 'documents-snapshot-3.npy', vectors)
    (bge / 'documents-snapshot-3-ids.json').write_text(json.dumps(doc_ids))
    for name, ids, values in (('test', test_ids, test_q), ('training', train_ids, train_q)):
        order = sorted(range(len(ids)), key=lambda i: ids[i], reverse=True)
        np.save(bge / f'queries-{name}.npy', values[order])
        (bge / f'queries-{name}-ids.json').write_text(json.dumps([ids[i] for i in order]))
    return tmp_path, bge


def test_it_writes_what_the_frozen_preparation_writes(frozen):
    root, bge = frozen
    summary = bge_sample.prepare(bge, root / 'qwen', root / 'bge-sample', pinned=None)
    assert summary['exact_full_sort_check']['passed']
    for name in ('vectors.npy', 'cohort.npy', 'lexical-rank.npy', 'queries.npy', 'exact-top100.npy',
                 'confirm-queries.npy', 'confirm-exact-top100.npy'):
        assert np.array_equal(np.load(root / 'qwen' / name), np.load(root / 'bge-sample' / name)), name
    for name in ('query-ids.json', 'confirm-query-ids.json'):
        assert (root / 'qwen' / name).read_text() == (root / 'bge-sample' / name).read_text()
    assert gzip.open(root / 'qwen' / 'doc-ids.txt.gz', 'rt').read() == gzip.open(root / 'bge-sample' / 'doc-ids.txt.gz', 'rt').read()
    theirs, mine = json.loads((root / 'qwen' / 'sample.json').read_text()), summary
    assert {k: theirs[k] for k in ('count', 'cohort_sizes', 'queries', 'metric', 'ids_sha256', 'confirm_queries')} == \
           {k: mine[k] for k in ('count', 'cohort_sizes', 'queries', 'metric', 'ids_sha256', 'confirm_queries')}


def test_it_refuses_another_document_order_other_queries_or_unpinned_files(frozen):
    root, bge = frozen
    ids = json.loads((bge / 'documents-snapshot-3-ids.json').read_text())
    (bge / 'documents-snapshot-3-ids.json').write_text(json.dumps(ids[1:] + ids[:1]))
    with pytest.raises(ValueError, match='document order'):
        bge_sample.prepare(bge, root / 'qwen', root / 'a', pinned=None)
    (bge / 'documents-snapshot-3-ids.json').write_text(json.dumps(ids))
    (bge / 'queries-test-ids.json').write_text(json.dumps(['t0', 't1', 't2', 't3', 't4', 'x']))
    with pytest.raises(ValueError, match='test query IDs'):
        bge_sample.prepare(bge, root / 'qwen', root / 'b', pinned=None)
    with pytest.raises(ValueError, match='verified BGE file'):
        bge_sample.prepare(bge, root / 'qwen', root / 'c')
