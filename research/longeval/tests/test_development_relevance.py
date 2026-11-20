
from research.longeval import development_relevance as development
from research.longeval.evaluation import write_run
from research.longeval.release import checksum
from research.longeval.storage import atomic_json


def test_development_only_sources_and_independent_parity(tmp_path, monkeypatch):
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    atomic_json(inputs / 'query-inputs.json', {})
    roles = {'development': ['dev'], 'calibration': ['cal']}
    monkeypatch.setattr(development, 'validate_inputs', lambda folder: ({'roles': roles}, []))
    exact = tmp_path / 'exact'; exact.mkdir()
    for precision in ('bfloat16', 'float32'):
        write_run(exact / f'{precision}.run', {'dev': [('a', 2.), ('b', 1.)], 'cal': [('b', 2.), ('a', 1.)]})
    atomic_json(exact / 'report.json', {'status': 'completed', 'snapshot': 1,
        'query_inputs_sha256': checksum(inputs / 'query-inputs.json'),
        'artifacts': {f'{p}.run': checksum(exact / f'{p}.run') for p in ('bfloat16', 'float32')}})
    entries = {}
    for family in ('raw', 'dctr'):
        path = tmp_path / f'{family}.qrels'
        path.write_text('dev 0 a 1\ndev 0 b 0\ncal 0 b 1\n')
        entries[f'longeval_sci_training_2026_qrels_{family}'] = {'cache_path': path.name, 'expected_md5': checksum(path, 'md5')}
    monkeypatch.setattr(development, 'manifest_entries', lambda folder: entries)
    result = development.evaluate_development(tmp_path, inputs, exact, tmp_path / 'out')
    assert result['query_ids'] == ['dev'] and result['status'] == 'completed'
    for value in result['sources'].values():
        assert value['evaluator_parity_passed']
        assert set(value['results']['float32']['per_query']) == {'dev'}
        assert value['results']['float32']['mean']['nDCG@10'] == 1
