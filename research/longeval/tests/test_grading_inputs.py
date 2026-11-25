import csv
import gzip
import json

import numpy as np
import pytest

from research.longeval import coverage, grading_inputs, judge


@pytest.fixture
def order(tmp_path):
    """Two builds of four documents' worth of top-10s for two queries, with their sample metadata."""
    meta = tmp_path / 'meta'
    meta.mkdir()
    docs = [f'd{i}' for i in range(30)]
    with gzip.open(meta / 'doc-ids.txt.gz', 'wt', encoding='utf-8') as handle:
        handle.write('\n'.join(docs) + '\n')
    (meta / 'query-ids.json').write_text(json.dumps(['q1', 'q2']))
    exact = np.tile(np.arange(100), (2, 1)).astype(np.int32)            # exact top 10: d0..d9 for both
    np.save(meta / 'exact-top100.npy', exact)
    builds = tmp_path / 'builds'
    for name, top in (('epzero/appended-11', [0, 1, 2, 3, 4, 5, 6, 7, 8, 20]),     # d20 substitutes
                      ('faiss_hnsw/reversed-11', [0, 1, 2, 3, 4, 5, 6, 7, 21, 22])):  # d21, d22 substitute
        folder = builds / name
        folder.mkdir(parents=True)
        labels = np.full((2, 100), -1, dtype=np.int32)
        labels[:, :10] = top
        np.save(folder / 'labels-ef200.npy', labels)
    ledger = tmp_path / 'ledger.csv'
    with ledger.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['query', 'document', 'snapshot', 'pooled_set'])
        writer.writeheader()
        writer.writerow({'query': 'q1', 'document': 'd20', 'snapshot': 3, 'pooled_set': '+extension'})
        writer.writerow({'query': 'q2', 'document': 'd21', 'snapshot': 3, 'pooled_set': 'other'})   # not a primary set
    return tmp_path, builds, meta, ledger


def test_substitutions_are_top_ten_documents_missing_from_exact_search(order):
    _, builds, meta, ledger = order
    found, read = grading_inputs.substitutions(builds, meta)
    assert read == ['epzero/appended-11', 'faiss_hnsw/reversed-11']
    assert found == {(q, d) for q in ('q1', 'q2') for d in ('d20', 'd21', 'd22')}
    assert grading_inputs.pooled(ledger) == {('q1', 'd20')}


def test_the_folder_holds_only_pairs_outside_the_primary_pool_in_the_frozen_format(order, monkeypatch):
    root, builds, meta, ledger = order
    release = root / 'release'
    (release / 'files').mkdir(parents=True)
    (release / 'files' / judge.TEST_FILE).write_text('')
    monkeypatch.setattr(grading_inputs, 'read_queries', lambda path: {'q1': 'graph search', 'q2': 'crop yields'})
    from pathlib import Path
    monkeypatch.setattr(grading_inputs, 'checksum', lambda path, *a: judge.TEST_SHA256 if Path(path).name == judge.TEST_FILE
                        else coverage.checksum(path, *a))
    from research.longeval import heldout_analysis
    monkeypatch.setattr(heldout_analysis, 'load_test_qrels', lambda path: ({judge.FAMILY: {}}, None))
    monkeypatch.setattr(judge, 'official_pairs', lambda qrels: {('q2', 'd22'): (1, 3)})
    monkeypatch.setattr(judge, 'document_texts', lambda folder, wanted: {d: f'text of {d}' for d in wanted})
    contract = grading_inputs.write_substitutions(release, ledger, builds, meta, root / 'out')
    assert contract['pairs'] == 5 and contract['substitutions'] == 6 and contract['already_in_primary_pool'] == 1
    rows = judge._read_jsonl(root / 'out' / 'pairs.jsonl')
    assert sorted((r['query_id'], r['doc_id']) for r in rows) == sorted(
        {(q, d) for q in ('q1', 'q2') for d in ('d20', 'd21', 'd22')} - {('q1', 'd20')})
    key = {r['pair']: r for r in judge._read_jsonl(root / 'out' / 'key.jsonl')}
    assert not any(r['validation'] for r in key.values())
    assert [key[r['pair']]['official_grade'] for r in rows if (r['query_id'], r['doc_id']) == ('q2', 'd22')] == [1]
    coverage.read_inputs(root / 'out')                                   # the frozen grading's check
    universe = json.loads((root / 'out' / 'universe.json').read_text())
    assert universe['snapshots']['1'] == {} and sum(len(v) for v in universe['snapshots']['3'].values()) == 5
