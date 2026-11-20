import json
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from research.longeval import trajectory
from research.longeval.backends import Backend
from research.longeval.oracle import exact_mapped
from research.longeval.release import checksum
from research.longeval.tests.test_mapped import snapshot_fixture, ARCHIVE_SHA


def small_problem(tmp_path):
    rng = np.random.default_rng(83)
    vectors = rng.normal(size=(48, 16)).astype('<f4')
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    q = rng.normal(size=(10, 16)).astype('<f4')
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    roles = {'calibration': SimpleNamespace(ids=[f'q{i}' for i in range(4)], vectors=q[:4]),
             'development': SimpleNamespace(ids=[f'q{i}' for i in range(4, 10)], vectors=q[4:])}
    specs, references = [], []
    for step, count in enumerate((20, 32, 48), 1):
        order = rng.permutation(count)
        ids = [f'd{i:03d}' for i in order]
        folder = snapshot_fixture(tmp_path / f'input-{step}', ids, vectors[order])
        specs.append({'snapshot': step, 'folder': folder, 'archive_sha256': ARCHIVE_SHA,
                      'vector_sha256': checksum(folder / 'vecs.f4')})
        references.append(exact_mapped(ids, vectors[order], [f'q{i}' for i in range(10)], q, k=10))
    config = {'schema': 1, 'M': 4, 'ef_construction': 32, 'k': 10, 'ef': 20,
              'ef_grid': [10, 20, 40], 'recall_target': .95, 'threads': 1, 'batch_rows': 5}
    return specs, roles, references, config


@pytest.mark.parametrize('kind', ['epzero', 'faiss_hnsw'])
def test_one_index_trajectories_replay_equal_and_shuffled_explicit(tmp_path, monkeypatch, kind):
    specs, queries, refs, config = small_problem(tmp_path)
    live = weakref.WeakSet()
    counts = []
    def allocate(*args, **kwargs):
        assert not live, 'an earlier index is still retained'
        backend = Backend(*args, **kwargs)
        live.add(backend); counts.append(1)
        return backend
    monkeypatch.setattr(trajectory, 'Backend', allocate)
    reports = {}
    for mode in ('incremental', 'replay', 'shuffled'):
        reports[mode] = trajectory.trajectory_core(specs, queries, refs, tmp_path / mode, config,
            kind=kind, mode=mode, seed=11, provenance_record={'synthetic_fixture': True})
        assert not live
        assert reports[mode]['status'] == 'completed'
    assert len(counts) == 7  # one persistent index, then three fresh instances per rebuild mode
    state_key = 'topology_sha256' if kind == 'epzero' else 'serialized_index_sha256'
    for a, b in zip(reports['incremental']['snapshots'], reports['replay']['snapshots']):
        assert a['state'][state_key] == b['state'][state_key]
        for regime in ('frozen', 'matched_recall'):
            assert a['runs'][regime]['sha256'] == b['runs'][regime]['sha256']
    assert reports['incremental']['snapshots'][1]['state']['parent_state_sha256'] is not None
    assert all(s['state']['parent_state_sha256'] is None for s in reports['replay']['snapshots'])
    assert reports['shuffled']['snapshots'][0]['state']['insertion_order_sha256'] != reports['replay']['snapshots'][0]['state']['insertion_order_sha256']


def test_corrupt_snapshot_fails_before_index_build(tmp_path):
    specs, queries, refs, config = small_problem(tmp_path)
    path = specs[0]['folder'] / 'vecs.f4'
    raw = bytearray(path.read_bytes()); raw[0] ^= 1; path.write_bytes(raw)
    with pytest.raises(ValueError, match='vector bytes changed'):
        trajectory.trajectory_core(specs, queries, refs, tmp_path / 'result', config,
            kind='epzero', mode='incremental', seed=11, provenance_record={})
    assert json.loads((tmp_path / 'result/report.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('kind', ['epzero', 'faiss_hnsw'])
def test_mapped_insertion_only_gathers_bounded_batches(kind):
    base = np.eye(12, dtype='<f4')
    sizes = []
    class Bounded:
        def __getitem__(self, rows):
            sizes.append(len(rows))
            assert len(rows) <= 3
            return base[rows]
    table = SimpleNamespace(ids=[str(i) for i in range(12)], vectors=Bounded())
    backend = Backend(kind, {'M': 4, 'ef_construction': 32}, {'dim': 12, 'metric': 'ip'}, table.ids, 11)
    trajectory.insert_mapped(backend, table, list(reversed(table.ids)), 3)
    assert sizes == [3, 3, 3, 3]
    assert len(backend.live) == 12
    with pytest.raises(ValueError, match='duplicate'):
        trajectory.insert_mapped(backend, table, ['1'], 3)
