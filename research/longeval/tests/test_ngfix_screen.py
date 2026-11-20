import csv
import json

import numpy as np
import pytest

from research.longeval.ngfix_screen import VARIANTS, GRID, check_inputs, fixture, verify_results
from research.longeval.ngfix_inputs import read_matrix, write_matrix
from research.longeval.release import checksum
from research.longeval.storage import atomic_json


@pytest.fixture
def evidence(tmp_path):
    """Fabricated verifier inputs only; these are never native research results."""
    inputs, output = tmp_path / 'inputs', tmp_path / 'output'
    fixture(inputs)
    output.mkdir()
    _, info, scores = check_inputs(inputs)
    with (output / 'rankings.tsv').open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['variant', 'query', 'ef', 'rank', 'id', 'distance'])
        for variant in VARIANTS:
            for qi, q in enumerate(info['order']):
                for ef in GRID:
                    order = np.lexsort((np.arange(scores.shape[1]), -scores[qi]))
                    # All development conditions are perfect, but calibration ef100 fails.
                    if qi < 8 and ef == 100: order = order[100:]
                    for rank, at in enumerate(order[:100], 1):
                        writer.writerow([variant, qi, ef, rank, at, 1-scores[qi, at]])
    with (output / 'queries.tsv').open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['variant', 'query', 'ef', 'repeat', 'seconds', 'upstream_ndc'])
        for variant in VARIANTS:
            for qi in range(16):
                for ef in GRID:
                    for repeat in range(3): writer.writerow([variant, qi, ef, repeat, .001, 400])
    (output / 'base.index').write_bytes(b'fake-state-base')
    with (output / 'stages.tsv').open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['variant', 'stage', 'seconds', 'serialized_bytes'])
        writer.writerow(['base', 'build', .01, len(b'fake-state-base')])
        for variant in VARIANTS:
            raw = b'fake-state-' + variant.encode()
            for phase in ('before', 'after'): (output / f'{variant}.{phase}.index').write_bytes(raw)
            writer.writerow([variant, 'repair', .01, len(raw)])
    return inputs, output


def test_full_verification_and_selection_uses_only_calibration(evidence):
    inputs, output = evidence
    record = verify_results(inputs, output)
    assert record['query_conditions'] == 256 and record['timed_calls'] == 768
    assert record['scored_pairs_checked'] == 25600
    assert all(r['selected_ef'] == 200 and r['development']['mean_recall'] == 1 for r in record['selected'])
    atomic_json(output / 'verification.json', record)
    returned = verify_results(inputs, output, check_states=False)
    assert returned['states_checked_here'] is False


@pytest.mark.parametrize('filename,old,new,match', [
    ('rankings.tsv', '\t1\t', '\t0\t', 'ranking'),
    ('queries.tsv', '0.001', 'nan', 'timing'),
    ('stages.tsv', '0.01', '-1', 'stage'),
])
def test_rejects_malformed_native_outputs(evidence, filename, old, new, match):
    inputs, output = evidence
    path = output / filename
    path.write_text(path.read_text().replace(old, new, 1))
    with pytest.raises(ValueError, match=match): verify_results(inputs, output)


def test_rejects_wrong_scores_even_with_valid_ids(evidence):
    inputs, output = evidence
    path = output / 'rankings.tsv'
    lines = path.read_text().splitlines()
    parts = lines[1].split('\t'); parts[-1] = '0'
    lines[1] = '\t'.join(parts)
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError, match='scores'): verify_results(inputs, output)


def test_rejects_missing_timing_repeat(evidence):
    inputs, output = evidence
    path = output / 'queries.tsv'
    path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
    with pytest.raises(ValueError, match='coverage'): verify_results(inputs, output)


def test_rejects_query_time_graph_mutation(evidence):
    inputs, output = evidence
    (output / 'rfix.after.index').write_bytes(b'changed')
    with pytest.raises(ValueError, match='graph state'): verify_results(inputs, output)


def test_rejects_mispaired_calibration_groundtruth_after_rehash(evidence):
    inputs, _ = evidence
    path = inputs / 'calibration_top200.ibin'
    rows = read_matrix(path, '<i4'); rows[[0, 1]] = rows[[1, 0]]
    temporary = inputs / 'changed.ibin'
    write_matrix(temporary, rows, '<i4')
    path.write_bytes(temporary.read_bytes())
    report = json.loads((inputs / 'report.json').read_text())
    report['artifacts'][path.name] = checksum(path)
    atomic_json(inputs / 'report.json', report)
    with pytest.raises(ValueError, match='mispaired'): check_inputs(inputs)


def test_rejects_returned_file_changed_after_node_verification(evidence):
    inputs, output = evidence
    record = verify_results(inputs, output)
    atomic_json(output / 'verification.json', record)
    path = output / 'queries.tsv'
    path.write_text(path.read_text().replace('0.001', '0.002', 1))
    with pytest.raises(ValueError, match='returned artifact'): verify_results(inputs, output, check_states=False)
