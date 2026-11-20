from types import SimpleNamespace

import numpy as np
import pytest

from research.longeval import heldout_analysis as analysis
from research.longeval import ivf_trajectory as ivf
from research.longeval import trajectory
from research.longeval.heldout import reference_agreement
from research.longeval.tests.test_ivf_trajectory import config as ivf_config
from research.longeval.tests.test_trajectory import small_problem


def as_evaluation(queries):
    return {'calibration': queries['calibration'], 'evaluation': queries['development']}


@pytest.mark.parametrize('kind', ['epzero', 'faiss_hnsw'])
def test_evaluation_role_reproduces_the_development_path(tmp_path, kind):
    specs, queries, refs, config = small_problem(tmp_path)
    dev = trajectory.trajectory_core(specs, queries, refs, tmp_path / 'dev', config, kind=kind,
                                     mode='incremental', seed=11, provenance_record={})
    held = trajectory.trajectory_core(specs, as_evaluation(queries), refs, tmp_path / 'held', config, kind=kind,
                                      mode='incremental', seed=11, provenance_record={}, evaluation_role='evaluation')
    assert held['evaluation_role'] == 'evaluation' and 'evaluation_role' not in dev
    for a, b in zip(dev['snapshots'], held['snapshots']):
        untimed = lambda c: {**c, 'points': [{k: v for k, v in p.items() if k != 'wall_seconds'} for p in c['points']]}
        assert a['state'] == b['state'] and untimed(a['calibration']) == untimed(b['calibration'])
        for regime in ('frozen', 'matched_recall'):
            assert a['runs'][regime]['sha256'] == b['runs'][regime]['sha256']
            assert b['runs'][regime]['evaluation_query_count'] == a['runs'][regime]['development_query_count']
    with pytest.raises(ValueError, match='query roles'):
        trajectory.trajectory_core(specs, queries, refs, tmp_path / 'bad', config, kind=kind, mode='incremental',
                                   seed=11, provenance_record={}, evaluation_role='evaluation')


def test_ivf_evaluation_role_reproduces_the_development_path(tmp_path):
    specs, queries, refs, _ = small_problem(tmp_path)
    dev = ivf.ivf_core(specs, queries, refs, tmp_path / 'dev', ivf_config(), mode='frozen_incremental', seed=11,
                       provenance_record={})
    held = ivf.ivf_core(specs, as_evaluation(queries), refs, tmp_path / 'held', ivf_config(), mode='frozen_incremental',
                        seed=11, provenance_record={}, evaluation_role='evaluation')
    for a, b in zip(dev['snapshots'], held['snapshots']):
        assert a['serialized_index_sha256'] == b['serialized_index_sha256']
        for regime in a['runs']:
            assert a['runs'][regime]['sha256'] == b['runs'][regime]['sha256']
            assert b['runs'][regime]['file'].endswith(f'evaluation.{regime}.run')
    with pytest.raises(ValueError, match='unknown evaluation role'):
        ivf.ivf_core(specs, queries, refs, tmp_path / 'x', ivf_config(), mode='frozen_incremental', seed=11,
                     provenance_record={}, evaluation_role='test')


def synthetic(degraded):
    """Exact ranks each query's relevant document first; degraded arms lose it at snapshot 3 for half the queries."""
    queries = [f'q{i:02d}' for i in range(20)]
    relevant = {q: {f'r{q}': 1, f'n{q}': 0} for q in queries}
    exact = {s: {q: [(f'r{q}', 2.), (f'n{q}', 1.)] for q in queries} for s in analysis.SNAPSHOTS}
    runs = {}
    for backend, mode, seed in analysis.expected_conditions():
        regimes = analysis.IVF_REGIMES if backend == 'faiss_ivf' else analysis.HNSW_REGIMES
        for step in analysis.SNAPSHOTS:
            for regime in regimes:
                lose = backend in degraded and step == 3
                runs[(backend, mode, seed, step, regime)] = {
                    q: ([(f'n{q}', 1.), (f'x{q}', .5)] if lose and i < 10 else exact[step][q]) for i, q in enumerate(queries)}
    qrels = {'gpt-oss-120b': {s: relevant for s in analysis.SNAPSHOTS}, 'human': {3: relevant}}
    return runs, exact, qrels, queries


def test_primary_contrast_holm_and_margin_decisions():
    runs, exact, qrels, queries = synthetic({'epzero'})
    result, rows = analysis.analyze_core(runs, exact, qrels, queries, primary_family='gpt-oss-120b',
                                         families=('gpt-oss-120b', 'human'))
    primary = result['primary']
    assert list(primary) == ['epzero_hnsw', 'faiss_hnsw', 'ivf_frozen']
    assert primary['epzero_hnsw']['mean_difference'] == pytest.approx(-.5)
    assert primary['epzero_hnsw']['label'] == 'changed' and primary['epzero_hnsw']['holm_p'] < .05
    for name in ('faiss_hnsw', 'ivf_frozen'):
        assert primary[name]['mean_difference'] == 0 and primary[name]['percentile_interval'] == [0, 0]
        assert primary[name]['label'] == 'negligible' and primary[name]['holm_p'] == 1
    assert result['panels']['gpt-oss-120b']['count'] == 20
    human = result['secondary']['human']['epzero_hnsw']
    assert list(human['gap_by_snapshot']) == [3] and 'gap_change_D' not in human
    assert human['gap_by_snapshot'][3]['mean_difference'] == pytest.approx(-.5)
    assert result['ann']['epzero incremental 11 3 matched_recall']['zero_recall_queries'] == 0
    assert result['exact_trend']['gpt-oss-120b']['nDCG@10_mean'] == {1: 1., 2: 1., 3: 1.}
    assert {r['family'] for r in rows} == {'gpt-oss-120b', 'human'}


def test_primary_requires_all_three_arms():
    runs, exact, qrels, queries = synthetic(set())
    runs = {k: v for k, v in runs.items() if k[0] != 'faiss_ivf'}
    with pytest.raises(ValueError, match='all three primary arms'):
        analysis.analyze_core(runs, exact, qrels, queries, primary_family='gpt-oss-120b', families=('gpt-oss-120b',))


def test_decision_rule_boundaries():
    assert analysis.decision([-.01, .01], .5)['label'] == 'negligible'
    assert analysis.decision([-.0101, .002], .5)['label'] == 'inconclusive'
    assert analysis.decision([-.03, -.011], .01)['label'] == 'changed'
    assert analysis.decision([.001, .009], .001) == {**analysis.decision([.001, .009], .001), 'label': 'negligible',
                                                     'holm_significant': True}


def test_panel_requires_a_positive_at_every_judged_snapshot():
    qrels = {1: {'a': {'d': 1}, 'b': {'d': 1}}, 2: {'a': {'d': 2}, 'b': {'d': 0}}}
    assert analysis.panel(qrels, ['a', 'b', 'c']) == ['a']


def test_equality_checks_report_any_difference():
    state = {'topology_sha256': 'x', 'parent_state_sha256': 'p', 'state_sha256': 's'}
    def report(topology):
        return {'snapshots': [{'snapshot': s, 'state': {**state, 'topology_sha256': topology}} for s in (1, 2, 3)]}
    run = {'q': [('d', 1.)]}
    reports = {('epzero', 'incremental', 11): report('x'), ('epzero', 'replay', 11): report('x')}
    runs = {(b, m, 11, s, r): run for b, m, _ in reports for s in (1, 2, 3) for r in analysis.HNSW_REGIMES}
    assert analysis.equality_checks(reports, runs)['all_identical']
    reports[('epzero', 'replay', 11)] = report('y')
    checked = analysis.equality_checks(reports, runs)
    assert not checked['all_identical'] and checked['comparisons'] == 3


def test_collect_requires_every_condition(tmp_path):
    with pytest.raises(ValueError, match='missing evidence'):
        analysis.collect([], 'evaluation', ['q'], {s: {'q': []} for s in (1, 2, 3)})


def test_reference_agreement_tolerates_rounding_only():
    left = {'q': [('a', 1.), ('b', .5)]}
    assert reference_agreement(left, {'q': [('a', 1. + 1e-12), ('b', .5)]})['passed']
    assert not reference_agreement(left, {'q': [('a', 1.), ('c', .5)]})['passed']
    assert not reference_agreement(left, {'q': [('a', 1.001), ('b', .5)]})['passed']


def test_exact_evaluation_on_a_synthetic_snapshot(tmp_path, monkeypatch):
    import json
    from research.longeval import heldout
    from research.longeval.evaluation import read_run
    from research.longeval.oracle import exact_mapped
    from research.longeval.release import checksum
    from research.longeval.tests.test_mapped import ARCHIVE_SHA, snapshot_fixture
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(150, 8)).astype('<f4')
    ids = [f'd{i:03d}' for i in range(150)]
    folder = snapshot_fixture(tmp_path / 'snap', ids, vectors)
    q = rng.normal(size=(6, 8)).astype('<f4')
    queries = {'calibration': SimpleNamespace(ids=['c0', 'c1'], vectors=q[:2]),
               'evaluation': SimpleNamespace(ids=['t0', 't1', 't2', 't3'], vectors=q[2:])}
    vector_sha = checksum(folder / 'vecs.f4')
    dev_run = exact_mapped(ids, vectors, ['c0', 'c1'], q[:2], k=100, block_rows=7)
    inputs = tmp_path / 'inputs'
    (inputs / 'training/exact/snapshot-1').mkdir(parents=True)
    (inputs / 'training/exact/snapshot-1/report.json').write_text('{}')
    scan = tmp_path / 'scan.json'
    scan.write_text(json.dumps({'status': 'completed', 'numeric_vectors_valid': True, 'vector_sha256': vector_sha}))
    monkeypatch.setattr(heldout, 'ARTIFACTS', ({'archive_sha256': ARCHIVE_SHA},) * 3)
    monkeypatch.setattr(heldout, 'load_heldout', lambda *a, **k: (queries, None, None,
                        {'test_inputs_sha256': 'i', 'test_encoding_sha256': 'e'}))
    monkeypatch.setattr(heldout, '_development_reference', lambda *a: ({'vector_sha256': vector_sha}, dev_run))
    report = heldout.exact_evaluation(folder, scan, inputs, tmp_path / 'out', snapshot=1, block_rows=16)
    assert report['status'] == 'completed' and report['calibration_recheck']['passed']
    run = read_run(tmp_path / 'out/evaluation.run')
    assert run == exact_mapped(ids, vectors, queries['evaluation'].ids, q[2:], k=100)
    assert report['artifacts']['evaluation.run'] == checksum(tmp_path / 'out/evaluation.run')
    monkeypatch.setattr(heldout, '_development_reference', lambda *a: ({'vector_sha256': 'other'}, dev_run))
    with pytest.raises(ValueError, match='reference vectors differ'):
        heldout.exact_evaluation(folder, scan, inputs, tmp_path / 'out2', snapshot=1)
