import json

import pytest

from research.longeval.evaluation import write_run
from research.longeval.ivf_trajectory import MODES, ivf_core
from research.longeval.ivf_verification import verify_experiment
from research.longeval.tests.test_ivf_trajectory import config
from research.longeval.tests.test_trajectory import small_problem


@pytest.fixture
def experiment(tmp_path):
    specs, queries, refs, _ = small_problem(tmp_path)
    for step, ref in enumerate(refs, 1):
        folder = tmp_path / 'references' / f'snapshot-{step}'
        folder.mkdir(parents=True)
        write_run(folder / 'float32.run', ref)
    output = tmp_path / 'experiment'
    for mode in MODES:
        ivf_core(specs, queries, refs, output / mode, config(), mode=mode, seed=11,
                 provenance_record={'development_inputs': {'synthetic': True}})
    return output, tmp_path / 'references', {role: table.ids for role, table in queries.items()}


def test_complete_fixture_verifies(experiment):
    report = verify_experiment(*experiment)
    assert report['status'] == 'verified' and len(report['rows']) == 27
    assert report['same_training_rebuild_equal']


@pytest.mark.parametrize('change,error', [('work', 'work count'), ('selection', 'selected probe'),
                                        ('population', 'population'), ('state', 'rebuild control')])
def test_corrupted_evidence_rejected(experiment, change, error):
    output, _, _ = experiment
    path = output / 'frozen_rebuild/report.json'
    report = json.loads(path.read_text())
    stage = report['snapshots'][2]
    if change == 'work':
        stage['runs']['all_lists']['queries'][0]['scanned_vectors'] += 1
    elif change == 'selection':
        stage['selection']['target_met'] = not stage['selection']['target_met']
    elif change == 'population':
        stage['runs']['all_lists']['queries'][0]['query_id'] = 'unknown'
    else:
        stage['serialized_index_sha256'] = '0' * 64
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=error):
        verify_experiment(*experiment)
