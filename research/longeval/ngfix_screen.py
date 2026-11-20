"""Checked native NGFix integration screen, with independent score verification."""
import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

import numpy as np

from .ngfix_inputs import read_matrix, write_matrix
from .release import checksum
from .storage import atomic_json, ingest_dense

VARIANTS = ('base', 'ngfix', 'rfix', 'ngfix_rfix')
GRID = (100, 200, 400, 800)
COMMIT = '9acd987cf1c420069f8fd602c0d4e4aa81053558'


def package_verify(package):
    package = Path(package).resolve()
    manifest = json.loads((package / 'manifest.json').read_text())
    if manifest['upstream_commit'] != COMMIT:
        raise ValueError('upstream commit differs')
    for name, expected in manifest['files'].items():
        path = (package / name).resolve()
        if not path.is_relative_to(package) or not path.is_file() or checksum(path) != expected:
            raise ValueError(f'package identity mismatch: {name}')
    return manifest


def check_inputs(folder):
    folder = Path(folder)
    record = json.loads((folder / 'report.json').read_text())
    for name, expected in record['artifacts'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or checksum(path) != expected:
            raise ValueError('input artifact identity mismatch')
    data = read_matrix(folder / 'base.fbin', '<f4')
    queries = read_matrix(folder / 'queries.fbin', '<f4')
    gt = read_matrix(folder / 'calibration_top200.ibin', '<i4')
    info = json.loads((folder / 'queries.json').read_text())
    labels = np.asarray(json.loads((folder / 'labels.json').read_text()), dtype=np.int64)
    cal, dev = info['roles']['calibration'], info['roles']['development']
    if (not cal or not dev or len(set(cal+dev)) != len(cal+dev) or info['order'] != cal+dev
            or len(labels) != len(data) or len(set(labels)) != len(labels)
            or list(data.shape) != record['base_shape'] or len(queries) != record['query_count']
            or len(queries) != len(cal+dev) or data.shape[1] != queries.shape[1]
            or gt.shape != (len(cal), 200) or len(cal) != record['calibration_count']
            or set(info['exact100']) != set(cal+dev)):
        raise ValueError('input roles or dimensions differ')
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(queries)):
        raise ValueError('nonfinite vectors')
    # Independent float64 reference validation for all exposed queries, never test queries.
    scores = queries.astype(np.float64) @ data.astype(np.float64).T
    for qi, qid in enumerate(info['order']):
        order = np.lexsort((labels, -scores[qi]))[:200]
        if list(map(int, order[:100])) != info['exact100'][qid]:
            raise ValueError('exact reference differs')
        if qi < len(cal) and not np.array_equal(gt[qi], order):
            raise ValueError('calibration top200 differs or is mispaired')
    return record, info, scores


def fixture(folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=False)
    rng = np.random.default_rng(20261119)
    data = rng.normal(size=(512, 64)).astype(np.float32)
    queries = rng.normal(size=(16, 64)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    scores = queries.astype(np.float64) @ data.astype(np.float64).T
    neighbors = np.asarray([np.lexsort((np.arange(len(data)), -row))[:200] for row in scores], dtype='<i4')
    ids = [f'fixture-{i}' for i in range(16)]
    write_matrix(folder / 'base.fbin', data, '<f4')
    write_matrix(folder / 'queries.fbin', queries, '<f4')
    write_matrix(folder / 'calibration_top200.ibin', neighbors[:8], '<i4')
    atomic_json(folder / 'queries.json', {'order': ids, 'roles': {'calibration': ids[:8], 'development': ids[8:]},
        'exact100': {qid: list(map(int, row[:100])) for qid, row in zip(ids, neighbors)}})
    atomic_json(folder / 'labels.json', list(range(512)))
    atomic_json(folder / 'report.json', {'status': 'prepared', 'synthetic': True, 'base_shape': [512, 64],
        'query_count': 16, 'calibration_count': 8, 'held_out_retrieval_evaluated': False,
        'artifacts': {p.name: checksum(p) for p in folder.iterdir()}})
    check_inputs(folder)


def sample_from_dense(dense, template, output):
    """Reconstruct exactly the already verified sample; no index build or fitting here."""
    dense, template, output = map(Path, (dense, template, output))
    expected = json.loads((template / 'report.json').read_text())
    ingestion = json.loads((dense / 'ingestion.json').read_text())
    pinned = json.loads((template / 'snapshot-1-ingestion.json').read_text())
    if ingestion != pinned or ingestion['status'] != 'verified':
        raise ValueError('snapshot ingestion differs from pinned local evidence')
    path = dense / 'vecs.f4'
    if checksum(path) != pinned['members']['vecs.f4']['sha256']:
        raise ValueError('snapshot vectors differ')
    labels = np.asarray(json.loads((template / 'labels.json').read_text()), dtype=np.int64)
    if len(labels) != 8192 or labels.min() < 0 or labels.max() >= 869902:
        raise ValueError('invalid pinned sample rows')
    matrix = np.memmap(path, mode='r', dtype='<f4', shape=(869902, 2560))
    selected = np.asarray(matrix[labels], dtype='<f4')
    output.mkdir(exist_ok=False)
    for name in ('queries.fbin', 'calibration_top200.ibin', 'queries.json', 'labels.json', 'report.json'):
        shutil.copyfile(template / name, output / name)
    write_matrix(output / 'base.fbin', selected, '<f4')
    if checksum(output / 'base.fbin') != expected['artifacts']['base.fbin']:
        raise ValueError('reconstructed sample differs')
    check_inputs(output)


def table(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def verify_results(inputs, output, *, check_states=True):
    inputs, output = map(Path, (inputs, output))
    input_record, info, scores = check_inputs(inputs)
    nq = len(info['order'])
    conditions = {(v, q, ef) for v in VARIANTS for q in range(nq) for ef in GRID}
    ranks, timings = defaultdict(list), defaultdict(dict)
    for row in table(output / 'rankings.tsv'):
        key = (row['variant'], int(row['query']), int(row['ef']))
        ranks[key].append((int(row['rank']), int(row['id']), float(row['distance'])))
    if set(ranks) != conditions:
        raise ValueError('ranking condition coverage differs')
    max_error = 0.
    recalls = {}
    for key, values in ranks.items():
        values.sort()
        if ([r for r, _, _ in values] != list(range(1, 101)) or len({i for _, i, _ in values}) != 100
                or any(i < 0 or i >= scores.shape[1] or not math.isfinite(d) for _, i, d in values)):
            raise ValueError('invalid ranking')
        if any(values[i][2] > values[i+1][2] for i in range(99)):
            raise ValueError('ranking distances not ordered')
        error = max(abs(d-(1-scores[key[1], i])) for _, i, d in values)
        if error > 2e-6:
            raise ValueError('native scores differ from independent float64 scores')
        max_error = max(max_error, error)
        truth = set(info['exact100'][info['order'][key[1]]])
        recalls[key] = len(truth & {i for _, i, _ in values}) / 100
    for row in table(output / 'queries.tsv'):
        key = (row['variant'], int(row['query']), int(row['ef']))
        repeat, seconds, ndc = int(row['repeat']), float(row['seconds']), int(row['upstream_ndc'])
        if (key not in conditions or repeat not in (0, 1, 2) or repeat in timings[key]
                or not math.isfinite(seconds) or seconds <= 0 or not 0 <= ndc < scores.shape[1]):
            raise ValueError('invalid timing condition or counter')
        timings[key][repeat] = (seconds, ndc)
    if set(timings) != conditions or any(set(v) != {0, 1, 2} for v in timings.values()):
        raise ValueError('timing coverage differs')
    if any(len({ndc for _, ndc in v.values()}) != 1 for v in timings.values()):
        raise ValueError('distance counter changed across repeats')
    stage_rows = table(output / 'stages.tsv')
    expected_stages = {('base', 'build')} | {(v, 'repair') for v in VARIANTS}
    if len(stage_rows) != 5 or {(r['variant'], r['stage']) for r in stage_rows} != expected_stages:
        raise ValueError('stage coverage differs')
    for row in stage_rows:
        if not math.isfinite(float(row['seconds'])) or float(row['seconds']) < 0 or int(row['serialized_bytes']) <= 0:
            raise ValueError('invalid stage cost')
    if check_states:
        states = {}
        for variant in VARIANTS:
            before, after = output / f'{variant}.before.index', output / f'{variant}.after.index'
            if checksum(before) != checksum(after):
                raise ValueError('querying changed graph state')
            states[variant] = {'sha256': checksum(before), 'bytes': before.stat().st_size}
        if checksum(output / 'base.index') != states['base']['sha256']:
            raise ValueError('base state changed on reload')
        for row in stage_rows:
            if int(row['serialized_bytes']) != states[row['variant']]['bytes']:
                raise ValueError('serialized size differs from stage record')
    else:
        saved = json.loads((output / 'verification.json').read_text())
        states = saved['states']
        for name, expected in saved['artifacts'].items():
            if checksum(output / name) != expected:
                raise ValueError('returned artifact differs from node verification')
    rows, selected = [], []
    for variant in VARIANTS:
        for role in ('calibration', 'development'):
            qindices = [info['order'].index(q) for q in info['roles'][role]]
            for ef in GRID:
                keys = [(variant, q, ef) for q in qindices]
                rows.append({'variant': variant, 'role': role, 'ef': ef,
                    'mean_recall': statistics.mean(recalls[k] for k in keys),
                    'minimum_recall': min(recalls[k] for k in keys),
                    'mean_query_median_seconds': statistics.mean(statistics.median(t for t, _ in timings[k].values()) for k in keys)})
        for target in (.95, .99):
            eligible = [r for r in rows if r['variant'] == variant and r['role'] == 'calibration' and r['mean_recall'] >= target]
            ef = min((r['ef'] for r in eligible), default=None)
            selected.append({'variant': variant, 'target': target, 'selected_ef': ef,
                'development': next((r for r in rows if r['variant'] == variant and r['role'] == 'development' and r['ef'] == ef), None)})
    report = {'status': 'verified', 'synthetic': input_record.get('synthetic', False),
        'query_conditions': len(conditions), 'timed_calls': 3*len(conditions), 'scored_pairs_checked': 100*len(conditions),
        'max_coordinate_score_error': max_error, 'states': states, 'groups': rows, 'selected': selected,
        'input_report_sha256': checksum(inputs / 'report.json'), 'source_sha256': checksum(__file__),
        'states_checked_here': check_states, 'held_out_retrieval_evaluated': False,
        'artifacts': {name: checksum(output / name) for name in ('queries.tsv', 'rankings.tsv', 'stages.tsv')},
        'limits': ['Exposed sample integration screen; not a novel-method or full-data performance result',
                   'Native C++ timings exclude Python binding overhead; no cross-binding speedup inferred',
                   'Upstream distance counter excludes initial entry scoring',
                   'Fifty fitting queries are not evidence against a properly trained competitor']}
    if not check_states:
        for key in ('query_conditions', 'timed_calls', 'scored_pairs_checked', 'states', 'groups', 'selected',
                    'input_report_sha256', 'artifacts'):
            if report[key] != saved[key]:
                raise ValueError(f'local result recheck differs: {key}')
        if abs(report['max_coordinate_score_error'] - saved['max_coordinate_score_error']) > 1e-12:
            raise ValueError('independent score error changed materially')
    return report


def execute(binary, inputs, output):
    check_inputs(inputs)
    start = time.monotonic()
    output = Path(output)
    log = output.with_suffix('.log')
    with log.open('x') as stream:
        subprocess.run([str(Path(binary).resolve()), str(Path(inputs)/'base.fbin'), str(Path(inputs)/'queries.fbin'),
            str(Path(inputs)/'calibration_top200.ibin'), str(output)], stdout=stream, stderr=subprocess.STDOUT, check=True)
    report = verify_results(inputs, output)
    if report['synthetic'] and any(r['mean_recall'] != 1. for r in report['groups'] if r['ef'] == 800):
        raise ValueError('exhaustive-width synthetic fixture failed exact recovery')
    report['execution'] = {'binary_sha256': checksum(binary), 'elapsed_including_verification_seconds': time.monotonic()-start,
        'platform': platform.platform(), 'cpuinfo': Path('/proc/cpuinfo').read_text().split('\n\n')[0],
        'affinity': sorted(os.sched_getaffinity(0)), 'numpy_version': np.__version__}
    atomic_json(output / 'verification.json', report)
    print(json.dumps({'status': 'verified', 'synthetic': report['synthetic'], 'conditions': report['query_conditions']}), flush=True)


def native_negative_checks(binary, inputs, work):
    """Fail-fast native loader checks on AVX-512 hardware, before any corpus download."""
    binary, inputs, work = map(Path, (binary, inputs, work))
    work.mkdir(exist_ok=False)
    gt = read_matrix(inputs / 'calibration_top200.ibin', '<i4')
    gt[0, 1] = gt[0, 0]
    write_matrix(work / 'duplicate.ibin', gt, '<i4')
    (work / 'truncated.fbin').write_bytes(b'bad')
    cases = [([], 'usage:'),
        ([str(work/'truncated.fbin'), str(inputs/'queries.fbin'), str(inputs/'calibration_top200.ibin'), str(work/'bad-header-output')], 'matrix header'),
        ([str(inputs/'base.fbin'), str(inputs/'queries.fbin'), str(work/'duplicate.ibin'), str(work/'bad-gt-output')], 'neighbor IDs')]
    evidence = []
    for arguments, diagnostic in cases:
        result = subprocess.run([str(binary.resolve()), *arguments], capture_output=True, text=True)
        if result.returncode != 1 or diagnostic not in result.stderr:
            raise ValueError('native rejection check failed')
        evidence.append({'expected': diagnostic, 'exit_code': result.returncode, 'stderr': result.stderr})
    atomic_json(work / 'verification.json', {'status': 'verified', 'cases': evidence, 'binary_sha256': checksum(binary)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    command = sub.add_parser('package-verify'); command.add_argument('package', type=Path)
    command = sub.add_parser('fixture'); command.add_argument('output', type=Path)
    command = sub.add_parser('sample')
    for name in ('dense', 'template', 'output'): command.add_argument(name, type=Path)
    command = sub.add_parser('download'); command.add_argument('dense', type=Path)
    command = sub.add_parser('execute')
    for name in ('binary', 'inputs', 'output'): command.add_argument(name, type=Path)
    command = sub.add_parser('native-negative-checks')
    for name in ('binary', 'inputs', 'work'): command.add_argument(name, type=Path)
    command = sub.add_parser('verify')
    for name in ('inputs', 'output'): command.add_argument(name, type=Path)
    command.add_argument('--returned', action='store_true')
    args = parser.parse_args()
    if args.action == 'package-verify':
        value = package_verify(args.package); print(f"Verified {len(value['files'])} packaged files")
    elif args.action == 'fixture': fixture(args.output)
    elif args.action == 'sample': sample_from_dense(args.dense, args.template, args.output)
    elif args.action == 'download':
        stage = Path(os.environ['LONGEVAL_NODE_STAGE']).resolve()
        if not args.dense.resolve().is_relative_to(stage): raise ValueError('download must stay inside node stage')
        ingest_dense(args.dense, snapshots=(1,))
    elif args.action == 'execute': execute(args.binary, args.inputs, args.output)
    elif args.action == 'native-negative-checks': native_negative_checks(args.binary, args.inputs, args.work)
    elif args.action == 'verify':
        value = verify_results(args.inputs, args.output, check_states=not args.returned)
        destination = args.output / ('local-verification.json' if args.returned else 'verification.json')
        atomic_json(destination, value); print(json.dumps({'status': value['status'], 'conditions': value['query_conditions']}))


if __name__ == '__main__': main()
