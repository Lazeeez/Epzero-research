import json
import numpy as np
import pytest

from research.longeval.representation import compare_arrays, compare_encoding
from research.longeval.release import checksum
from research.longeval.storage import atomic_json


def test_comparison_joins_ids_and_preserves_signed_zero():
    a = np.array([[1, -0.0], [0, 1]], dtype='<f4')
    b = np.array([[0, 1], [1, 0]], dtype='<f4')
    result = compare_arrays(['a', 'b'], a, ['b', 'a'], b)
    assert result['cosine_min'] == 1
    assert result['bitwise_equal_rows'] == 1
    assert result['max_coordinate_error'] == 0


@pytest.mark.parametrize('bad', [np.array([[0, 0]], dtype='<f4'),
                               np.array([[float('nan'), 0]], dtype='<f4'),
                               np.array([[2, 0]], dtype='<f4'), np.array([[1, 0]], dtype='f8')])
def test_comparison_rejects_invalid_vectors(bad):
    with pytest.raises(ValueError):
        compare_arrays(['a'], np.array([[1, 0]], dtype='<f4'), ['a'], bad)


def test_comparison_rejects_wrong_membership_and_duplicate_ids():
    value = np.array([[1, 0], [0, 1]], dtype='<f4')
    for ids in (['a', 'a'], ['a', 'c']):
        with pytest.raises(ValueError):
            compare_arrays(['a', 'b'], value, ids, value)


def test_receipts_bind_candidate_to_input_and_output(tmp_path):
    ref, enc = tmp_path / 'ref', tmp_path / 'enc'
    ref.mkdir(); enc.mkdir()
    (ref / 'records.jsonl').write_text(json.dumps({'id': 'a', 'text': 'original'}) + '\n')
    for folder in (ref, enc):
        atomic_json(folder / 'ids.json', ['a'])
        np.save(folder / 'vectors.npy', np.array([[1, 0]], dtype='<f4'))
    atomic_json(ref / 'reference.json', {'status': 'completed', 'files': {
        name: checksum(ref / name) for name in ('records.jsonl', 'ids.json', 'vectors.npy')}})
    receipt = {'status': 'completed', 'dtype': 'bfloat16', 'device': 'cpu',
               'input_sha256': checksum(ref / 'records.jsonl'),
               'vectors_sha256': checksum(enc / 'vectors.npy'), 'ids_sha256': checksum(enc / 'ids.json')}
    atomic_json(enc / 'encoding.json', receipt)
    result = compare_encoding(ref, enc, tmp_path / 'result.json')
    assert result['bitwise_equal_rows'] == 1 and result['acceptance_threshold'] is None
    receipt['input_sha256'] = '0' * 64
    atomic_json(enc / 'encoding.json', receipt)
    with pytest.raises(ValueError, match='different input'):
        compare_encoding(ref, enc, tmp_path / 'bad.json')
    receipt['input_sha256'] = checksum(ref / 'records.jsonl')
    atomic_json(enc / 'encoding.json', receipt)
    np.save(enc / 'vectors.npy', np.array([[0, 1]], dtype='<f4'))
    with pytest.raises(ValueError, match='checksum'):
        compare_encoding(ref, enc, tmp_path / 'bad.json')


def test_preparation_checks_sample_and_joins_reordered_reference(tmp_path, monkeypatch):
    from research.longeval import dense
    from research.longeval.representation import prepare_sample
    from research.longeval.tests.test_mapped import snapshot_fixture, ARCHIVE_SHA

    monkeypatch.setattr(dense, 'ARTIFACTS', [{'archive_sha256': ARCHIVE_SHA}])
    folder = snapshot_fixture(tmp_path / 'snapshot', ['b', 'a'], [[0, 1], [1, 0]])
    sample = tmp_path / 'snapshot-1-encoder-sample.jsonl'
    sample.write_text(json.dumps({'id': 'a', 'text': 'first'}) + '\n' +
                      json.dumps({'id': 'b', 'text': 'second'}) + '\n')
    audit = tmp_path / 'audit.json'
    atomic_json(audit, {'status': 'completed', 'snapshots': [{'snapshot': 1,
                'conflicting_duplicate_rows': 0, 'encoder_sample_sha256': checksum(sample),
                'archive_sha256': 'text-archive', 'dense_coverage': {'missing_text_ids': 0,
                'id_map_sha256': checksum(folder / 'docnos.npids')}}]})
    out = tmp_path / 'bundle'
    assert prepare_sample(folder, audit, out)['count'] == 2
    np.testing.assert_array_equal(np.load(out / 'vectors.npy'), [[1, 0], [0, 1]])
    sample.write_text(sample.read_text().replace('first', 'changed'))
    with pytest.raises(ValueError, match='audited selection'):
        prepare_sample(folder, audit, tmp_path / 'bad')
