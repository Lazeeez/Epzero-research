import json

import numpy as np
import pytest

from research.longeval import query_diagnostic as query
from research.longeval.release import checksum
from research.longeval.storage import atomic_json
from research.longeval.query_sensitivity import ranking_changes


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    release = tmp_path / 'release'; release.mkdir()
    source = release / 'train.tsv'
    source.write_text(''.join(f'q{i:03d}\t query {i} with trailing space \n' for i in range(100)))
    baseline = tmp_path / 'baseline.py'; baseline.write_text('# pinned fixture')
    monkeypatch.setattr(query, 'BASELINE_SHA256', checksum(baseline))
    monkeypatch.setattr(query, 'manifest_entries', lambda path: {
        query.TRAIN: {'cache_path': 'train.tsv', 'expected_md5': checksum(source, 'md5')}})
    folder = tmp_path / 'prepared'
    query.prepare_queries(release, baseline, folder)
    return folder


@pytest.fixture
def fake_encoder(monkeypatch):
    def encode(model, records, output, *, dtype, device, threads, max_length):
        output.mkdir()
        rows = query.encoder.read_records(records)
        ids = [r['id'] for r in rows]
        values = np.zeros((100, 2560), dtype='<f4'); values[:, 0] = 1
        np.save(output / 'vectors.npy', values)
        atomic_json(output / 'ids.json', ids)
        report = {'status': 'completed', 'dtype': dtype, 'device': device, 'threads': threads,
                  'max_length': max_length, 'batch_size': 1, 'pooling': 'last', 'normalization': 'L2',
                  'attention': 'eager', 'truncation': 'reject', 'gpu': 'test', 'tf32': False,
                  'source_sha256': 'encoder-fixture', 'packages': {'torch': 'test'},
                  'model_identity': {'model': query.encoder.MODEL, 'revision': query.encoder.REVISION},
                  'input_sha256': checksum(records), 'vectors_sha256': checksum(output / 'vectors.npy'),
                  'ids_sha256': checksum(output / 'ids.json'), 'rows': [{'id': q, 'tokens': 10} for q in ids]}
        atomic_json(output / 'encoding.json', report)
        return report
    monkeypatch.setattr(query.encoder, 'encode_documents', encode)


def test_prompt_and_roles_preserve_text_without_qrels(prepared):
    contract, rows = query.validate_inputs(prepared)
    assert rows[0]['text'] == query.PROMPT + ' query 0 with trailing space '
    assert query.PROMPT.endswith('\nQuery:')
    assert len(contract['roles']['development']) == len(contract['roles']['calibration']) == 50
    assert not set(contract['roles']['development']) & set(contract['roles']['calibration'])
    assert not contract['approved_for_primary_experiments']


def test_input_tampering_is_rejected(prepared):
    p = prepared / 'records.jsonl'; p.write_text(p.read_text().replace('query 0', 'changed'))
    with pytest.raises(ValueError, match='checksum'):
        query.validate_inputs(prepared)


def test_different_prompt_rejected_even_with_updated_file_hash(prepared):
    p = prepared / 'records.jsonl'; p.write_text(p.read_text().replace('Query:', 'Query: '))
    report = json.loads((prepared / 'query-inputs.json').read_text())
    report['files']['records.jsonl'] = checksum(p)
    atomic_json(prepared / 'query-inputs.json', report)
    with pytest.raises(ValueError):
        query.validate_inputs(prepared)


def test_encoding_pair_roundtrip_and_setting_guard(prepared, fake_encoder, tmp_path):
    a, b = tmp_path / 'bf16', tmp_path / 'fp32'
    query.encode_queries('unused', prepared, a, dtype='bfloat16')
    query.encode_queries('unused', prepared, b, dtype='float32')
    result = query.compare_precisions(prepared, a, b, tmp_path / 'comparison.json')
    assert result['count'] == 100 and result['bitwise_equal_rows'] == 100
    raw = json.loads((b / 'encoder/encoding.json').read_text()); raw['tf32'] = True
    atomic_json(b / 'encoder/encoding.json', raw)
    receipt = json.loads((b / 'query-encoding.json').read_text())
    receipt['files']['encoder/encoding.json'] = checksum(b / 'encoder/encoding.json')
    atomic_json(b / 'query-encoding.json', receipt)
    with pytest.raises(ValueError, match='another setting'):
        query.load_pair(prepared, a, b)


def test_encoding_failure_records_failed_status(prepared, monkeypatch, tmp_path):
    def fail(*args, **kwargs): raise RuntimeError('no model')
    monkeypatch.setattr(query.encoder, 'encode_documents', fail)
    with pytest.raises(RuntimeError):
        query.encode_queries('unused', prepared, tmp_path / 'failed')
    assert json.loads((tmp_path / 'failed/query-encoding.json').read_text())['status'] == 'failed'


def test_exact_ranking_sensitivity_distinguishes_order_and_membership():
    original = [(str(i), float(20-i)) for i in range(10)]
    swapped = [(str(i), float(20-i)) for i in range(2, 10)] + [('1', 20.), ('0', 19.)]
    result = ranking_changes({'q': original}, {'q': swapped}, depth=10)['q']
    assert result['overlap_at_10'] == 1 and not result['ordered_top10_same'] and not result['top1_same']
    replaced = original[:-1] + [('new', 0.)]
    assert ranking_changes({'q': original}, {'q': replaced}, depth=10)['q']['overlap_at_10'] == .9
    with pytest.raises(ValueError):
        ranking_changes({'q': original}, {'q': original[:-1]}, depth=10)


def test_bounded_reference_pipeline_uses_verified_vectors(prepared, fake_encoder, tmp_path, monkeypatch):
    from research.longeval import query_sensitivity as sensitivity
    from research.longeval.tests.test_mapped import snapshot_fixture, ARCHIVE_SHA
    a, b = tmp_path / 'bf16', tmp_path / 'fp32'
    query.encode_queries('unused', prepared, a, dtype='bfloat16')
    query.encode_queries('unused', prepared, b, dtype='float32')
    vectors = np.zeros((120, 2560), dtype='<f4'); vectors[:, 0] = 1
    folder = snapshot_fixture(tmp_path / 'snapshot', [str(i) for i in range(120)], vectors)
    monkeypatch.setattr(sensitivity, 'ARTIFACTS', [{'archive_sha256': ARCHIVE_SHA}])
    scan = tmp_path / 'scan.json'
    atomic_json(scan, {'status': 'completed', 'numeric_vectors_valid': True,
                      'vector_sha256': checksum(folder / 'vecs.f4')})
    out = tmp_path / 'ranks'
    report = sensitivity.sensitivity(folder, scan, prepared, a, b, out, block_rows=37)
    assert report['status'] == 'completed'
    assert report['cohorts']['all_training']['mean']['overlap_at_100'] == 1
    assert report['cohorts']['development']['count'] == 50
    assert (out / 'bfloat16.run').read_bytes() == (out / 'float32.run').read_bytes()
    vectors[0, 0] = -1
    vectors.tofile(folder / 'vecs.f4')
    with pytest.raises(ValueError, match='differ from audit'):
        sensitivity.sensitivity(folder, scan, prepared, a, b, tmp_path / 'corrupt')
    assert json.loads((tmp_path / 'corrupt/report.json').read_text())['status'] == 'failed'
