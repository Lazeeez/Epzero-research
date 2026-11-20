import copy
import json

import pytest

from research.longeval import trajectory
from research.longeval.heldout_extension import SweepObserver
from research.longeval.tests.test_heldout import as_evaluation
from research.longeval.tests.test_trajectory import small_problem


@pytest.mark.parametrize('kind', ['epzero', 'faiss_hnsw'])
def test_sweep_reproduces_baseline_and_keeps_state(tmp_path, kind):
    specs, queries, refs, config = small_problem(tmp_path)
    queries = as_evaluation(queries)
    baseline = trajectory.trajectory_core(specs, queries, refs, tmp_path / 'base', config, kind=kind, mode='incremental',
                                          seed=11, provenance_record={}, evaluation_role='evaluation')
    observer = SweepObserver(tmp_path / 'inc/sweep', baseline, efs=(10, 20, 40))
    report = trajectory.trajectory_core(specs, queries, refs, tmp_path / 'inc/trajectory', config, kind=kind,
                                        mode='incremental', seed=11, provenance_record={}, snapshot_observer=observer,
                                        evaluation_role='evaluation')
    assert all(s['diagnostic']['baseline_reproduced'] for s in report['snapshots'])
    sweep = json.loads((tmp_path / 'inc/sweep/snapshot-3/report.json').read_text())
    assert list(sweep['runs']) == ['10', '20', '40'] and sweep['status'] == 'completed'
    # The ef-20 sweep run equals the primary fixed-ef run (config ef is 20).
    assert sweep['runs']['20']['sha256'] == baseline['snapshots'][2]['runs']['frozen']['sha256']
    shuffled = trajectory.trajectory_core(specs, queries, refs, tmp_path / 'shu/trajectory', config, kind=kind,
                                          mode='shuffled', seed=11, provenance_record={},
                                          snapshot_observer=SweepObserver(tmp_path / 'shu/sweep', efs=(10, 40)),
                                          evaluation_role='evaluation')
    assert shuffled['status'] == 'completed'
    changed = copy.deepcopy(baseline)
    changed['snapshots'][0]['state']['count'] += 1
    with pytest.raises(ValueError, match='did not reproduce'):
        trajectory.trajectory_core(specs, queries, refs, tmp_path / 'bad/trajectory', config, kind=kind,
                                   mode='incremental', seed=11, provenance_record={},
                                   snapshot_observer=SweepObserver(tmp_path / 'bad/sweep', changed, efs=(10,)),
                                   evaluation_role='evaluation')


def test_sweep_grid_must_increase(tmp_path):
    with pytest.raises(ValueError, match='increasing'):
        SweepObserver(tmp_path / 'x', efs=(200, 100))
