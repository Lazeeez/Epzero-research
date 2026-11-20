"""Role-leakage regression tests; fabricated outputs are not research results."""
import csv

import pytest

from research.longeval.ngfix_disjoint import prepare, protocol, read, save, select_settings, validate_roles, verify
from research.longeval.ngfix_inputs import read_matrix
from research.longeval.ngfix_screen import GRID, VARIANTS, check_inputs, fixture, verify_results


@pytest.fixture
def prepared(tmp_path):
    parent, inputs = tmp_path / 'parent', tmp_path / 'inputs'
    fixture(parent)
    prepare(parent, 0, inputs)
    return parent, inputs


def test_roles_disjoint_deterministic_and_native_sees_only_fit(prepared):
    parent, inputs = prepared
    plan = protocol(parent)
    assert plan == protocol(parent)
    original = read(parent / 'queries.json')
    for fold in plan['folds']:
        roles = fold['roles']
        fit, tune, dev = map(set, (roles['fitting'], roles['tuning'], roles['development']))
        assert not (fit & tune or fit & dev or tune & dev)
        assert fit | tune == set(original['roles']['calibration'])
        assert roles['development'] == original['roles']['development']
    roles, info = validate_roles(parent, inputs)
    assert len(read_matrix(inputs / 'calibration_top200.ibin', '<i4')) == len(roles['fitting'])
    assert info['order'][:len(roles['fitting'])] == roles['fitting']
    assert not (inputs / 'protocol.json').read_bytes().count(b'\r\n')


def test_rejects_reassigned_tuning_query_even_with_valid_json(prepared):
    parent, inputs = prepared
    record = read(inputs / 'protocol.json')
    roles = record['contract']['folds'][0]['roles']
    roles['fitting'][0], roles['tuning'][0] = roles['tuning'][0], roles['fitting'][0]
    save(inputs / 'protocol.json', record)
    with pytest.raises(ValueError, match='contract'):
        validate_roles(parent, inputs)


@pytest.mark.parametrize('fold', [-1, 3, True])
def test_rejects_invalid_fold(prepared, tmp_path, fold):
    parent, _ = prepared
    with pytest.raises(ValueError, match='fold'):
        prepare(parent, fold, tmp_path / 'bad')


def groups():
    return [{'variant': v, 'role': role, 'ef': ef,
             'mean_recall': 1. if role == 'fitting' else (.96 if role == 'tuning' and ef >= 200 else .90)}
            for v in VARIANTS for role in ('fitting', 'tuning', 'development') for ef in GRID]


def test_selection_ignores_perfect_fit_and_failed_development():
    rows = groups()
    selected = select_settings(rows)
    assert all(r['selected_ef'] == 200 and not r['development_target_met'] for r in selected if r['target'] == .95)
    assert all(r['selected_ef'] is None and r['development'] is None for r in selected if r['target'] == .99)
    for row in rows:
        if row['role'] != 'tuning': row['mean_recall'] = 1 - row['mean_recall']
    assert [r['selected_ef'] for r in select_settings(rows)] == [r['selected_ef'] for r in selected]


def test_selection_rejects_duplicate_role_and_nan():
    rows = groups()
    with pytest.raises(ValueError, match='coverage'): select_settings(rows + [rows[0]])
    rows[0]['mean_recall'] = float('nan')
    with pytest.raises(ValueError, match='recall'): select_settings(rows)


def fake_native(inputs, output):
    """Make exact scores with failed tuning ef100, perfect fitting and development."""
    output.mkdir()
    roles = read(inputs / 'protocol.json')['contract']['folds'][0]['roles']
    _, info, scores = check_inputs(inputs)
    with (output / 'rankings.tsv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['variant', 'query', 'ef', 'rank', 'id', 'distance'])
        for v in VARIANTS:
            for qi, q in enumerate(info['order']):
                for ef in GRID:
                    ordered = sorted(range(scores.shape[1]), key=lambda i: (-scores[qi, i], i))
                    if q in roles['tuning'] and ef == 100: ordered = ordered[100:]
                    for rank, i in enumerate(ordered[:100], 1):
                        writer.writerow([v, qi, ef, rank, i, 1-scores[qi, i]])
    with (output / 'queries.tsv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['variant', 'query', 'ef', 'repeat', 'seconds', 'upstream_ndc'])
        for v in VARIANTS:
            for qi in range(len(info['order'])):
                for ef in GRID:
                    for repeat in range(3): writer.writerow([v, qi, ef, repeat, .001, 400])
    with (output / 'stages.tsv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['variant', 'stage', 'seconds', 'serialized_bytes'])
        writer.writerow(['base', 'build', .01, 4])
        (output / 'base.index').write_bytes(b'base')
        for v in VARIANTS:
            for phase in ('before', 'after'): (output / f'{v}.{phase}.index').write_bytes(v.encode())
            writer.writerow([v, 'repair', .01, len(v)])
    save(output / 'verification.json', verify_results(inputs, output))


def test_raw_rankings_select_on_tuning_and_returned_tampering_rejected(prepared, tmp_path):
    parent, inputs = prepared
    output = tmp_path / 'native'
    fake_native(inputs, output)
    # Legacy machinery would choose ef100 from perfect fitting. New selector must not.
    assert all(r['selected_ef'] == 100 for r in read(output / 'verification.json')['selected'])
    result = verify(parent, inputs, output)
    assert all(r['selected_ef'] == 200 for r in result['selected'])
    assert result['query_conditions'] == 256 and len(result['groups']) == 48
    save(output / 'disjoint-verification.json', result)
    assert verify(parent, inputs, output, returned=True) == result
    result['selected'][0]['selected_ef'] = 100
    save(output / 'disjoint-verification.json', result)
    with pytest.raises(ValueError, match='returned disjoint'):
        verify(parent, inputs, output, returned=True)
