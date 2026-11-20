from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from research.longeval import ivf_trajectory as ivf
from research.longeval.tests.test_trajectory import small_problem


def config():
    return dict(schema=1, nlist=2, training_rows=16, iterations=3, k=10,
                nprobe_grid=[1, 2], recall_target=.95, threads=1, batch_rows=5, search_repeats=2)


def test_frozen_rebuild_reproduces_incremental_and_all_lists_are_exhaustive(tmp_path):
    specs, queries, refs, _ = small_problem(tmp_path)
    reports = {mode: ivf.ivf_core(specs, queries, refs, tmp_path / mode, config(),
               mode=mode, seed=11, provenance_record={'synthetic': True}) for mode in ivf.MODES}
    for step in range(3):
        a = reports['frozen_incremental']['snapshots'][step]
        b = reports['frozen_rebuild']['snapshots'][step]
        assert a['serialized_index_sha256'] == b['serialized_index_sha256']
        assert a['training']['snapshot'] == b['training']['snapshot'] == 1
        for regime in a['runs']:
            assert a['runs'][regime]['sha256'] == b['runs'][regime]['sha256']
        fresh = reports['refreshed_rebuild']['snapshots'][step]
        assert fresh['training']['snapshot'] == step + 1
        for report in reports.values():
            stage = report['snapshots'][step]
            assert stage['runs']['all_lists']['ann_recall_mean'] == 1.
            for row in stage['runs']['all_lists']['queries']:
                assert row['scanned_vectors'] == stage['live_count']
                assert row['total_vector_comparisons'] == stage['live_count'] + 2
                assert len(row['native_call_nanoseconds']) == 2
    assert reports['frozen_incremental']['snapshots'][0]['serialized_index_sha256'] == reports['refreshed_rebuild']['snapshots'][0]['serialized_index_sha256']
    assert reports['refreshed_rebuild']['snapshots'][2]['training']['sample_ids_sha256'] != reports['frozen_incremental']['snapshots'][2]['training']['sample_ids_sha256']


def test_training_sample_and_vector_gather_are_order_independent_and_bounded():
    ids = [f'd{i}' for i in range(100)]
    selected = ivf.sample_ids(ids, 20, 11)
    assert selected == ivf.sample_ids(list(reversed(ids)), 20, 11)
    assert selected != ivf.sample_ids(ids, 20, 29)
    values = np.arange(400, dtype=np.float32).reshape(100, 4)
    class Bounded:
        shape = values.shape
        def __getitem__(self, rows):
            assert len(rows) <= 3
            return values[rows]
    table = SimpleNamespace(ids=ids, vectors=Bounded())
    assert np.array_equal(ivf.gather(table, selected, 3), values[[ids.index(q) for q in selected]])


def test_short_search_is_infeasible_and_never_drops_queries(tmp_path):
    faiss.omp_set_num_threads(1)
    quantizer = faiss.IndexFlatIP(2)
    quantizer.add(np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    index = faiss.IndexIVFFlat(quantizer, 2, 2, faiss.METRIC_INNER_PRODUCT)
    index.add_with_ids(np.array([[1., 0.]], dtype=np.float32), np.array([0], dtype=np.int64))
    queries = SimpleNamespace(ids=['empty', 'present'], vectors=np.array([[0., 1.], [1., 0.]], dtype=np.float32))
    run, detail = ivf.search_cohort(index, ['doc'], queries, 1, 1)
    assert set(run) == set(queries.ids)
    assert not detail['feasible'] and detail['incomplete_queries'] == ['empty']
    assert run['empty'] == []
    from research.longeval.ivf_verification import verify_point
    path = tmp_path / 'partial.run'
    ivf.export_run(path, run)
    detail.update(file=path.name, sha256=ivf.checksum(path), mean_recall=None)
    reference = {q: [('doc', 1.)] for q in queries.ids}
    assert verify_point(tmp_path, detail, reference, queries.ids, [1, 0], 1, repeats=1) is None
    with pytest.raises(ValueError, match='no feasible'):
        ivf.select_probe([dict(nprobe=1, feasible=False, mean_recall=None)], .95)


def test_target_miss_is_recorded_not_relabelled_success():
    points = [dict(nprobe=1, feasible=True, mean_recall=.7), dict(nprobe=2, feasible=True, mean_recall=.8)]
    chosen = ivf.select_probe(points, .95)
    assert chosen['nprobe'] == 2 and not chosen['target_met']


def test_independent_scores_reject_wrong_scores_and_exhaustive_misses():
    table = SimpleNamespace(ids=['a', 'b', 'c'], vectors=np.array([[1., 0.], [.5, 0.], [0., 1.]], dtype=np.float32))
    queries = {'development': SimpleNamespace(ids=['q'], vectors=np.array([[1., 0.]], dtype=np.float32))}
    ref = {'q': [('a', 1.)]}
    with pytest.raises(ValueError, match='score differs'):
        ivf.verify_returned_scores(table, queries, [({'q': [('a', .7)]}, False)], ref)
    with pytest.raises(ValueError, match='non-boundary'):
        ivf.verify_returned_scores(table, queries, [({'q': [('b', .5)]}, True)], ref)
    assert ivf.verify_returned_scores(table, queries, [(ref, True)], ref)['status'] == 'verified'


def test_actual_membership_rejects_duplicate_ids():
    quantizer = faiss.IndexFlatIP(2)
    quantizer.add(np.array([[1., 0.]], dtype=np.float32))
    index = faiss.IndexIVFFlat(quantizer, 2, 1, faiss.METRIC_INNER_PRODUCT)
    index.add_with_ids(np.array([[1., 0.], [1., 0.]], dtype=np.float32), np.array([0, 0], dtype=np.int64))
    with pytest.raises(ValueError, match='duplicate'):
        ivf.verify_membership(index, ['a', 'b'], {'a', 'b'})


def test_corrupt_vector_rejected_before_training(tmp_path):
    specs, queries, refs, _ = small_problem(tmp_path)
    path = specs[0]['folder'] / 'vecs.f4'
    raw = bytearray(path.read_bytes()); raw[0] ^= 1; path.write_bytes(raw)
    with pytest.raises(ValueError, match='vector bytes changed'):
        ivf.ivf_core(specs, queries, refs, tmp_path / 'out', config(), mode='frozen_incremental', seed=11, provenance_record={})


@pytest.mark.parametrize('key,value', [('nprobe_grid', [1]), ('threads', 2), ('training_rows', 1), ('batch_rows', 0)])
def test_invalid_settings(key, value):
    settings = config(); settings[key] = value
    with pytest.raises(ValueError):
        ivf.validate_settings(settings, 'frozen_incremental', 11)
