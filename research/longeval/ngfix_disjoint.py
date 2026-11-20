"""Three-role NGFix screen layered on the immutable, verified native adapter.

The original binary calls its first (fitting) rows 'calibration'. That wire name
is retained solely for compatibility. Only this module's tuning rows select ef.
No returned development result may influence role assignment or selection.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess

from .ngfix_inputs import read_matrix, write_matrix
from .ngfix_screen import GRID, VARIANTS, check_inputs, execute, table, verify_results
from .release import checksum

VERSION = 'longeval-ngfix-disjoint-v1'


def save(path, value):
    # Platform-independent bytes: input identity must not depend on CRLF defaults.
    Path(path).write_bytes((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def read(path):
    return json.loads(Path(path).read_text())


def protocol(parent):
    parent = Path(parent)
    info = read(parent / 'queries.json')
    pool, development = info['roles']['calibration'], info['roles']['development']
    if (len(pool) < 4 or not development or len(set(pool + development)) != len(pool + development)
            or info['order'] != pool + development):
        raise ValueError('invalid parent query roles')
    ordered = sorted(pool, key=lambda q: (hashlib.sha256((VERSION + '|' + q).encode()).hexdigest(), q))
    fit_count = 3 * len(pool) // 5
    folds = []
    for fold in range(3):
        offset = fold * len(pool) // 3
        rotated = ordered[offset:] + ordered[:offset]
        folds.append({'fold': fold, 'roles': {'fitting': rotated[:fit_count],
                     'tuning': rotated[fit_count:], 'development': development}})
    return {'version': VERSION, 'parent_input_report_sha256': checksum(parent / 'report.json'),
            'parent_queries_sha256': checksum(parent / 'queries.json'), 'folds': folds,
            'targets': [.95, .99], 'ef_grid': list(GRID),
            'selection_rule': 'smallest ef meeting mean recall on tuning only; null if none',
            'held_out_retrieval_evaluated': False,
            'limits': ['Exploratory reuse of 100 exposed queries; not confirmatory evaluation',
                       'Three overlapping splits are not independent datasets or graph seeds',
                       'Thirty fitting queries in the real sample; no fair-training-budget superiority claim']}


def prepare(parent, fold, output):
    parent, output = Path(parent), Path(output)
    parent_record, info, _ = check_inputs(parent)
    contract = protocol(parent)
    if type(fold) is not int or fold not in range(3):
        raise ValueError('invalid fold')
    roles = contract['folds'][fold]['roles']
    order = roles['fitting'] + roles['tuning'] + roles['development']
    queries = read_matrix(parent / 'queries.fbin', '<f4')
    references = read_matrix(parent / 'calibration_top200.ibin', '<i4')
    query_rows = [info['order'].index(q) for q in order]
    fitting_rows = [info['roles']['calibration'].index(q) for q in roles['fitting']]
    output.mkdir(exist_ok=False)
    try:
        os.link(parent / 'base.fbin', output / 'base.fbin')
    except OSError:
        shutil.copyfile(parent / 'base.fbin', output / 'base.fbin')
    shutil.copyfile(parent / 'labels.json', output / 'labels.json')
    write_matrix(output / 'queries.fbin', queries[query_rows], '<f4')
    # The binary receives reference rows ONLY for fitting, in the same order.
    write_matrix(output / 'calibration_top200.ibin', references[fitting_rows], '<i4')
    save(output / 'queries.json', {'order': order,
        'roles': {'calibration': roles['fitting'], 'development': roles['tuning'] + roles['development']},
        'exact100': {q: info['exact100'][q] for q in order}})
    save(output / 'report.json', {'status': 'prepared', 'synthetic': parent_record.get('synthetic', False),
        'base_shape': parent_record['base_shape'], 'query_count': len(order),
        'calibration_count': len(roles['fitting']), 'held_out_retrieval_evaluated': False,
        'wire_role_note': 'legacy calibration means fitting; selection is in disjoint-verification.json',
        'artifacts': {name: checksum(output / name) for name in
            ('base.fbin', 'queries.fbin', 'calibration_top200.ibin', 'queries.json', 'labels.json')}})
    save(output / 'protocol.json', {'contract': contract, 'fold': fold})
    validate_roles(parent, output)


def validate_roles(parent, inputs):
    parent, inputs = Path(parent), Path(inputs)
    declared = read(inputs / 'protocol.json')
    expected = protocol(parent)
    fold = declared['fold']
    if declared['contract'] != expected or type(fold) is not int or fold not in range(3):
        raise ValueError('role contract differs from frozen parent')
    roles = expected['folds'][fold]['roles']
    original = read(parent / 'queries.json')
    _, info, _ = check_inputs(inputs)
    order = roles['fitting'] + roles['tuning'] + roles['development']
    if (info['order'] != order or info['roles']['calibration'] != roles['fitting']
            or info['roles']['development'] != roles['tuning'] + roles['development']):
        raise ValueError('native fitting rows do not match disjoint roles')
    # Source identity must hold even if someone recomputes an altered input report.
    for name in ('base.fbin', 'labels.json'):
        if checksum(inputs / name) != checksum(parent / name):
            raise ValueError('parent sample identity changed')
    import numpy as np
    parent_queries = read_matrix(parent / 'queries.fbin', '<f4')
    expected_queries = parent_queries[[original['order'].index(q) for q in order]]
    if not np.array_equal(expected_queries, read_matrix(inputs / 'queries.fbin', '<f4')):
        raise ValueError('query identity/order changed')
    if info['exact100'] != original['exact100']:
        raise ValueError('exact references changed')
    return roles, info


def select_settings(groups, targets=(.95, .99)):
    """Consume declared aggregate rows; fitting/development never choose ef."""
    indexed = {(r['variant'], r['role'], r['ef']): r for r in groups}
    expected = {(v, role, ef) for v in VARIANTS for role in ('fitting', 'tuning', 'development') for ef in GRID}
    if len(indexed) != len(groups) or set(indexed) != expected:
        raise ValueError('role/variant/effort coverage differs')
    if any(not math.isfinite(r['mean_recall']) or not 0 <= r['mean_recall'] <= 1 for r in groups):
        raise ValueError('invalid recall')
    selected = []
    for variant in VARIANTS:
        for target in targets:
            if not 0 < target <= 1:
                raise ValueError('invalid recall target')
            eligible = [ef for ef in GRID if indexed[variant, 'tuning', ef]['mean_recall'] >= target]
            ef = min(eligible, default=None)
            selected.append({'variant': variant, 'target': target, 'selected_ef': ef,
                'tuning': indexed[variant, 'tuning', ef] if ef is not None else None,
                'development': indexed[variant, 'development', ef] if ef is not None else None,
                'development_target_met': indexed[variant, 'development', ef]['mean_recall'] >= target if ef is not None else None})
    return selected


def verify(parent, inputs, output, *, returned=False):
    parent, inputs, output = map(Path, (parent, inputs, output))
    roles, info = validate_roles(parent, inputs)
    # Unchanged verifier checks all IDs, float64 scores, repeats, counters and
    # graph hashes. Its legacy fit-based 'selected' field is intentionally unused.
    mechanical = verify_results(inputs, output, check_states=not returned)
    ranks, timings = defaultdict(list), defaultdict(list)
    for row in table(output / 'rankings.tsv'):
        ranks[row['variant'], int(row['query']), int(row['ef'])].append(int(row['id']))
    for row in table(output / 'queries.tsv'):
        timings[row['variant'], int(row['query']), int(row['ef'])].append(float(row['seconds']))
    groups = []
    for variant in VARIANTS:
        for role, ids in roles.items():
            indices = [info['order'].index(q) for q in ids]
            for ef in GRID:
                recalls = [len(set(ranks[variant, q, ef]) & set(info['exact100'][info['order'][q]]))/100 for q in indices]
                groups.append({'variant': variant, 'role': role, 'ef': ef, 'query_count': len(ids),
                    'mean_recall': statistics.mean(recalls), 'minimum_recall': min(recalls),
                    'mean_query_median_seconds': statistics.mean(statistics.median(timings[variant, q, ef]) for q in indices)})
    result = {'status': 'verified', 'synthetic': mechanical['synthetic'],
        'protocol': read(inputs / 'protocol.json'), 'groups': groups, 'selected': select_settings(groups),
        'protocol_sha256': checksum(inputs / 'protocol.json'),
        'input_report_sha256': checksum(inputs / 'report.json'),
        'source_sha256': checksum(__file__), 'mechanical_verifier_sha256': checksum(Path(__file__).with_name('ngfix_screen.py')),
        'query_conditions': mechanical['query_conditions'], 'timed_calls': mechanical['timed_calls'],
        'scored_pairs_checked': mechanical['scored_pairs_checked'], 'states': mechanical['states'],
        'artifacts': mechanical['artifacts'], 'held_out_retrieval_evaluated': False,
        'selection_uses_fitting_queries': False,
        'limits': read(inputs / 'protocol.json')['contract']['limits']}
    if returned:
        saved = read(output / 'disjoint-verification.json')
        for key, value in result.items():
            if saved.get(key) != value:
                raise ValueError(f'returned disjoint verification differs: {key}')
    return result


def run(binary, parent, inputs, output):
    validate_roles(parent, inputs)
    execute(binary, inputs, output)
    result = verify(parent, inputs, output)
    result['execution'] = {'cwd': os.getcwd(), 'affinity': sorted(os.sched_getaffinity(0)),
        'binary_sha256': checksum(binary),
        'mount_view': subprocess.run(['findmnt', '-T', str(Path(inputs).resolve()), '-o', 'SOURCE,FSTYPE,TARGET'],
                                     capture_output=True, text=True, check=True).stdout}
    save(Path(output) / 'disjoint-verification.json', result)
    print(json.dumps({'status': 'verified', 'fold': result['protocol']['fold'],
                      'selection': 'tuning only', 'conditions': result['query_conditions']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('protocol'); p.add_argument('parent', type=Path); p.add_argument('output', type=Path)
    p = sub.add_parser('prepare'); p.add_argument('parent', type=Path); p.add_argument('fold', type=int); p.add_argument('output', type=Path)
    p = sub.add_parser('execute')
    for name in ('binary', 'parent', 'inputs', 'output'): p.add_argument(name, type=Path)
    p = sub.add_parser('verify')
    for name in ('parent', 'inputs', 'output'): p.add_argument(name, type=Path)
    p.add_argument('--returned', action='store_true')
    args = parser.parse_args()
    if args.action == 'protocol': save(args.output, protocol(args.parent))
    elif args.action == 'prepare': prepare(args.parent, args.fold, args.output)
    elif args.action == 'execute': run(args.binary, args.parent, args.inputs, args.output)
    else:
        result = verify(args.parent, args.inputs, args.output, returned=args.returned)
        save(args.output / ('local-disjoint-verification.json' if args.returned else 'disjoint-verification.json'), result)
        print(json.dumps({'status': result['status'], 'conditions': result['query_conditions']}))


if __name__ == '__main__':
    main()
