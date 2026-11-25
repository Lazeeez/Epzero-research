import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from research.longeval import bge_heldout
from research.longeval.evaluation import rank
from research.longeval.heldout import ROLE

N, DIM = 400, 8


def _docnos(folder, ids):
    from npids import Lookup
    folder.mkdir(parents=True)
    with Lookup.build(ids, folder / 'docnos.npids'):
        pass
    (folder / 'pt_meta.json').write_text(json.dumps({'type': 'dense_index', 'format': 'flex', 'vec_size': 2560,
                                                     'doc_count': len(ids)}))


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    ids = [f'doc{i:04d}' for i in rng.permutation(N)]
    vectors = rng.standard_normal((N, DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    bge = tmp_path / 'bge'
    bge.mkdir()
    np.save(bge / 'documents-snapshot-3.npy', vectors)
    (bge / 'documents-snapshot-3-ids.json').write_text(json.dumps(ids))
    # The released snapshots grow; snapshots 1 and 2 have their own row orders.
    first, second = ids[:200], ids[:300]
    docnos = tmp_path / 'docnos'
    _docnos(docnos / 'snapshot-1', list(rng.permutation(first)))
    _docnos(docnos / 'snapshot-2', list(rng.permutation(second)))
    _docnos(docnos / 'snapshot-3', ids)
    test_ids, train_ids = [f't{i}' for i in range(6)], [f'r{i}' for i in range(5)]
    for name, qids in (('test', test_ids), ('training', train_ids)):
        q = rng.standard_normal((len(qids), DIM)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        np.save(bge / f'queries-{name}.npy', q[::-1])                     # stored in another order
        (bge / f'queries-{name}-ids.json').write_text(json.dumps(qids[::-1]))
    sample = tmp_path / 'sample'
    sample.mkdir()
    (sample / 'query-ids.json').write_text(json.dumps(test_ids))
    from research.longeval import query_diagnostic
    monkeypatch.setattr(query_diagnostic, 'validate_inputs',
                        lambda folder: ({'roles': {'calibration': train_ids[:3]}}, None))
    return tmp_path, bge, docnos, sample, vectors, ids, test_ids, train_ids


def test_the_frozen_reader_opens_the_bge_snapshots_in_the_released_row_order(stand_in):
    from research.longeval.mapped import MappedSnapshot
    root, bge, docnos, _, vectors, ids, *_ = stand_in
    specs = bge_heldout.write_snapshots(bge, docnos, root / 'out', pinned=None)
    assert [s['snapshot'] for s in specs] == [1, 2, 3]
    position = {d: i for i, d in enumerate(ids)}
    for spec in specs:
        with MappedSnapshot(spec['folder'], expected_archive_sha256=spec['archive_sha256']) as mapped:
            rows = [position[d] for d in mapped.ids]
            assert np.array_equal(np.asarray(mapped.vectors), vectors[rows])
    assert len({s['archive_sha256'] for s in specs}) == 3


def test_a_released_document_without_a_bge_vector_stops_it(stand_in):
    root, bge, docnos, _, _, ids, *_ = stand_in
    (bge / 'documents-snapshot-3-ids.json').write_text(json.dumps(ids[:-1] + ['other']))
    with pytest.raises(ValueError, match='no BGE vector'):
        bge_heldout.write_snapshots(bge, docnos, root / 'out', pinned=None)


def test_cohorts_and_references_follow_the_study(stand_in):
    root, bge, docnos, sample, vectors, ids, test_ids, train_ids = stand_in
    queries = bge_heldout.cohorts(bge, root / 'training', sample)
    assert queries['calibration'].ids == train_ids[:3] and queries[ROLE].ids == test_ids
    stored = np.load(bge / 'queries-test.npy')[::-1]
    assert np.array_equal(queries[ROLE].vectors, stored)
    specs = bge_heldout.write_snapshots(bge, docnos, root / 'out', pinned=None)
    refs = bge_heldout.references(specs, queries)
    present = [set(ids[:200]), set(ids[:300]), set(ids)]
    for step, ref in enumerate(refs):
        for q, v in zip(queries[ROLE].ids, queries[ROLE].vectors):
            scores = vectors.astype(np.float64) @ v.astype(np.float64)
            expected = rank([(d, float(scores[i])) for i, d in enumerate(ids) if d in present[step]])[:100]
            assert [d for d, _ in ref[q]] == [d for d, _ in expected]


def test_a_prepared_run_goes_through_the_frozen_hnsw_core(stand_in, monkeypatch):
    root, bge, docnos, sample, vectors, ids, test_ids, _ = stand_in
    queries = bge_heldout.cohorts(bge, root / 'training', sample)
    # A BGE sample's exact neighbours, as bge_sample writes them, for the cross-check.
    with gzip.open(sample / 'doc-ids.txt.gz', 'wt', encoding='utf-8') as handle:
        handle.write('\n'.join(ids) + '\n')
    lexical = np.empty(N, dtype=np.int64)
    lexical[np.argsort(np.array(ids, dtype=object), kind='stable')] = np.arange(N)
    scores = queries[ROLE].vectors.astype(np.float64) @ vectors.astype(np.float64).T
    np.save(sample / 'exact-top100.npy', np.stack([np.lexsort((-lexical, -s))[:100] for s in scores]).astype(np.int32))
    evidence = bge_heldout.prepare(bge, docnos, root / 'training', sample, root / 'prepared', pinned=None)
    assert evidence['sample_agreement']['passed'] and evidence['sample_agreement']['identical_id_order']
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(key, '1')
    pytest.importorskip('epzero')
    config = Path(bge_heldout.__file__).with_name('trajectory-development.json')
    bge_heldout.run_hnsw(root / 'prepared', root / 'run', config, kind='faiss_hnsw', mode='incremental', seed=11,
                         package_root=root, native_origin='test')
    report = json.loads((root / 'run' / 'report.json').read_text())
    assert report['provenance']['heldout_inputs']['bge_encoder'] == bge_heldout.IDENTITY
    assert report['evaluation_role'] == ROLE


def _job(stand_in, name):
    """One job's prepared folder and a report recording it, as a held-out job writes them."""
    root, bge, docnos, sample, *_ = stand_in
    job = root / 'jobs' / name
    bge_heldout.prepare(bge, docnos, root / 'training', sample, job / 'prepared', pinned=None)
    _, _, _, identity = bge_heldout.load_prepared(job / 'prepared')
    (job / 'trajectory').mkdir(parents=True)
    report = {'provenance': {'heldout_inputs': {k: identity[k] for k in ('bge_prepared_sha256', 'bge_references_sha256')}}}
    (job / 'trajectory' / 'report.json').write_text(json.dumps(report))
    return job


def _restamp(path, mtime):
    """The same references, gzipped at another time (the header's timestamp is all that changes)."""
    raw = gzip.open(path).read()
    with open(path, 'wb') as out, gzip.GzipFile(filename='', fileobj=out, mode='wb', mtime=mtime) as gz:
        gz.write(raw)


def test_inputs_are_checked_by_content_not_by_gzip_bytes_or_folders(stand_in):
    first, second = _job(stand_in, 'a'), _job(stand_in, 'b')
    for step in bge_heldout.SNAPSHOTS:                                   # job b wrote its files later
        _restamp(second / 'prepared' / f'references-snapshot-{step}.json.gz', 1_000_000 + step)
    evidence = json.loads((second / 'prepared' / 'prepared.json').read_text())
    evidence['references_sha256'] = {str(s): bge_heldout.sha256(second / 'prepared' / f'references-snapshot-{s}.json.gz')
                                     for s in bge_heldout.SNAPSHOTS}
    (second / 'prepared' / 'prepared.json').write_text(json.dumps(evidence, indent=2) + '\n')
    _, _, _, identity = bge_heldout.load_prepared(second / 'prepared')
    report = {'provenance': {'heldout_inputs': {k: identity[k] for k in ('bge_prepared_sha256', 'bge_references_sha256')}}}
    (second / 'trajectory' / 'report.json').write_text(json.dumps(report))
    assert (second / 'prepared' / 'prepared.json').read_bytes() != (first / 'prepared' / 'prepared.json').read_bytes()
    assert bge_heldout.content_fingerprint(first / 'prepared') == bge_heldout.content_fingerprint(second / 'prepared')
    reports = sorted((stand_in[0] / 'jobs').rglob('report.json'))
    assert bge_heldout.check_inputs(reports, first / 'prepared') == 2

    # A report that records another job's inputs, and a job whose references differ, both stop it.
    (second / 'trajectory' / 'report.json').write_text((first / 'trajectory' / 'report.json').read_text())
    with pytest.raises(ValueError, match="own job"):
        bge_heldout.check_inputs(reports, first / 'prepared')
    third = _job(stand_in, 'c')
    path = third / 'prepared' / 'references-snapshot-3.json.gz'
    refs = json.loads(gzip.open(path).read())
    query = next(iter(refs))
    refs[query] = refs[query][::-1]
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        json.dump(refs, handle)
    evidence = json.loads((third / 'prepared' / 'prepared.json').read_text())
    evidence['references_sha256']['3'] = bge_heldout.sha256(path)
    (third / 'prepared' / 'prepared.json').write_text(json.dumps(evidence, indent=2) + '\n')
    _, _, _, identity = bge_heldout.load_prepared(third / 'prepared')
    (third / 'trajectory' / 'report.json').write_text(json.dumps(
        {'provenance': {'heldout_inputs': {k: identity[k] for k in ('bge_prepared_sha256', 'bge_references_sha256')}}}))
    with pytest.raises(ValueError, match="differ in content"):
        bge_heldout.check_inputs([third / 'trajectory' / 'report.json'], first / 'prepared')
