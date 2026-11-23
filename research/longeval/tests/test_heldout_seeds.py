"""Further construction seeds: the calibration observer leaves the index unchanged and reproduces calibration."""
import json

import pytest

from research.longeval import heldout_seeds, trajectory
from research.longeval.tests.test_heldout import as_evaluation
from research.longeval.tests.test_trajectory import small_problem


@pytest.mark.parametrize('kind', ['epzero', 'faiss_hnsw'])
def test_observer_records_calibration_per_query_without_changing_the_run(tmp_path, kind):
    specs, queries, refs, config = small_problem(tmp_path)
    queries = as_evaluation(queries)
    plain = trajectory.trajectory_core(specs, queries, refs, tmp_path / 'plain', config, kind=kind, mode='incremental',
                                       seed=101, provenance_record={}, evaluation_role='evaluation')
    observed = trajectory.trajectory_core(specs, queries, refs, tmp_path / 'observed', config, kind=kind,
                                          mode='incremental', seed=101, provenance_record={},
                                          snapshot_observer=heldout_seeds.CalibrationObserver(),
                                          evaluation_role='evaluation')
    def untimed(calibration):
        return {**calibration, 'points': [{k: v for k, v in p.items() if k != 'wall_seconds'} for p in calibration['points']]}

    for a, b in zip(plain['snapshots'], observed['snapshots']):
        # The observer searches only: states, calibration and runs are those of the unobserved trajectory.
        assert a['state']['state_sha256'] == b['state']['state_sha256']
        assert untimed(a['calibration']) == untimed(b['calibration'])
        assert a['runs']['frozen']['sha256'] == b['runs']['frozen']['sha256']
        diagnostic = b['diagnostic']
        assert set(diagnostic['calibration_recall_per_query']) == {str(ef) for ef in config['ef_grid']}
        assert all(set(v) == set(queries['calibration'].ids) for v in diagnostic['calibration_recall_per_query'].values())
        assert diagnostic['calibration_recall_mean'][str(b['calibration']['effort'])] == b['calibration']['calibration_recall']
    assert json.loads((tmp_path / 'observed/report.json').read_text())['status'] == 'completed'


def test_only_the_recorded_seeds_are_accepted(tmp_path):
    with pytest.raises(ValueError, match='seed must be one of'):
        heldout_seeds.run_hnsw(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, kind='epzero', seed=11,
                               package_root=tmp_path, native_origin='test')
    with pytest.raises(ValueError, match='seed must be one of'):
        heldout_seeds.run_ivf_seed(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, seed=47,
                                   package_root=tmp_path, native_origin='test')
    assert len(set(heldout_seeds.EXTRA_SEEDS)) == 7 and not {11, 29, 47} & set(heldout_seeds.EXTRA_SEEDS)
