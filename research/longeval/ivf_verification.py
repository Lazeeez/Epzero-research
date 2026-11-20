"""Verify exported IVF controls independently of their calibration/search implementation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .contracts import digest
from .evaluation import read_run
from .release import checksum
from .storage import atomic_json


def verify_point(folder, point, reference, query_ids, sizes, k, *, repeats):
    path = folder / point['file']
    if checksum(path) != point['sha256']:
        raise ValueError('IVF run hash mismatch')
    run = read_run(path)
    # Entirely empty outputs are omitted by TREC serialization; report still retains the ID.
    if set(run) - set(query_ids):
        raise ValueError('IVF run has an unexpected query')
    rows = point['queries']
    if [r['query_id'] for r in rows] != list(query_ids):
        raise ValueError('IVF query population/order mismatch')
    values, missing = {}, []
    for row in rows:
        q = row['query_id']
        pairs = run.get(q, [])
        if len(pairs) != row['returned'] or len({d for d, _ in pairs}) != len(pairs) or len(pairs) > k:
            raise ValueError('IVF run count/duplicate mismatch')
        if len(pairs) < k:
            missing.append(q)
        probes = row['probed_lists']
        if len(probes) != point['nprobe'] or len(set(probes)) != len(probes) or any(type(p) is not int or not 0 <= p < len(sizes) for p in probes):
            raise ValueError('IVF probe list mismatch')
        expected = sum(sizes[p] for p in probes)
        if (row['scanned_vectors'] != expected or row['coarse_centroid_comparisons'] != len(sizes)
                or row['total_vector_comparisons'] != expected + len(sizes)):
            raise ValueError('IVF work count mismatch')
        if len(row['native_call_nanoseconds']) != repeats or any(type(t) is not int or t <= 0 for t in row['native_call_nanoseconds']):
            raise ValueError('IVF latency record mismatch')
        if len(reference[q]) != k:
            raise ValueError('exact reference depth mismatch')
        values[q] = len({d for d, _ in pairs} & {d for d, _ in reference[q]}) / k
    if missing != point['incomplete_queries'] or point['feasible'] != (not missing):
        raise ValueError('IVF feasibility record mismatch')
    if point['warmups_per_query'] != 1 or point['timed_repeats'] != repeats:
        raise ValueError('IVF repeat protocol mismatch')
    mean = math.fsum(values.values()) / len(query_ids) if not missing else None
    reported = point.get('mean_recall', point.get('ann_recall_mean'))
    if (mean is None) != (reported is None) or mean is not None and not math.isclose(mean, reported, abs_tol=1e-14, rel_tol=0):
        raise ValueError('IVF recall recount mismatch')
    if 'ann_recall_per_query' in point and values != point['ann_recall_per_query']:
        raise ValueError('IVF per-query recall mismatch')
    return mean


def verify_experiment(output, references, query_roles):
    output, references = Path(output), Path(references)
    reports, rows = {}, []
    artifacts = {}
    for mode in ('frozen_incremental', 'frozen_rebuild', 'refreshed_rebuild'):
        folder = output / mode
        report = json.loads((folder / 'report.json').read_text())
        reports[mode] = report
        if report['status'] != 'completed' or report['mode'] != mode or [s['snapshot'] for s in report['snapshots']] != [1, 2, 3]:
            raise ValueError('incomplete IVF trajectory')
        config = report['config']
        frozen = None
        for stage in report['snapshots']:
            step = stage['snapshot']
            reference = read_run(references / f'snapshot-{step}/float32.run')
            if set(reference) != set(query_roles['calibration'] + query_roles['development']):
                raise ValueError('IVF reference query population mismatch')
            sizes = stage['list_sizes']
            if len(sizes) != config['nlist'] or any(type(n) is not int or n < 0 for n in sizes) or sum(sizes) != stage['live_count']:
                raise ValueError('IVF list size mismatch')
            training = stage['training']
            expected_step = step if mode == 'refreshed_rebuild' else 1
            if training['snapshot'] != expected_step:
                raise ValueError('IVF training history mismatch')
            training_folder = folder / f'training-{expected_step}'
            for name, expected in training['files'].items():
                if checksum(training_folder / name) != expected:
                    raise ValueError('IVF training artifact mismatch')
            ids = json.loads((training_folder / 'ids.json').read_text())
            centroids = np.load(training_folder / 'centroids.npy', allow_pickle=False)
            import hashlib
            if (ids != sorted(set(ids)) or len(ids) != config['training_rows'] or digest(ids) != training['sample_ids_sha256']
                    or len(centroids) != config['nlist'] or not np.isfinite(centroids).all()
                    or hashlib.sha256(centroids).hexdigest() != training['centroids_sha256']):
                raise ValueError('IVF training identity mismatch')
            points = stage['calibration_grid']
            if [p['nprobe'] for p in points] != config['nprobe_grid']:
                raise ValueError('IVF calibration grid mismatch')
            for point in points:
                verify_point(folder, point, reference, query_roles['calibration'], sizes, config['k'], repeats=1)
            eligible = [p for p in points if p['feasible'] and p['mean_recall'] >= config['recall_target']]
            valid = [p for p in points if p['feasible']]
            chosen = min(eligible, key=lambda p: p['nprobe']) if eligible else max(valid, key=lambda p: (p['mean_recall'], -p['nprobe']))
            selection = stage['selection']
            if (selection['nprobe'] != chosen['nprobe'] or selection['target_met'] != bool(eligible)
                    or selection['target'] != config['recall_target']
                    or not math.isclose(selection['mean_calibration_recall'], chosen['mean_recall'], abs_tol=1e-14)):
                raise ValueError('IVF selected probe mismatch')
            frozen = selection['nprobe'] if frozen is None else frozen
            if stage['frozen_snapshot1_nprobe'] != frozen:
                raise ValueError('frozen probe changed')
            if set(stage['runs']) != {'frozen_snapshot1', 'current_calibration', 'all_lists'}:
                raise ValueError('IVF development regimes mismatch')
            for regime, point in stage['runs'].items():
                expected = {'frozen_snapshot1': frozen, 'current_calibration': selection['nprobe'], 'all_lists': config['nlist']}[regime]
                if point['nprobe'] != expected:
                    raise ValueError('development probe differs from selection')
                mean = verify_point(folder, point, reference, query_roles['development'], sizes, config['k'], repeats=config['search_repeats'])
                rows.append({'mode': mode, 'seed': report['seed'], 'snapshot': step, 'regime': regime,
                             'nprobe': expected, 'mean_recall': mean, 'feasible': point['feasible']})
            scored = stage['score_verification']
            if scored['status'] != 'verified' or not 0 <= scored['max_absolute_score_error'] <= 1e-5 or scored['absolute_tolerance'] != 1e-5:
                raise ValueError('independent score check failed')
    first = reports['frozen_incremental']
    for mode, report in reports.items():
        if report['config'] != first['config'] or report['seed'] != first['seed'] or report['provenance']['development_inputs'] != first['provenance']['development_inputs']:
            raise ValueError('IVF arms have different inputs or settings')
    for a, b in zip(first['snapshots'], reports['frozen_rebuild']['snapshots']):
        for key in ('serialized_index_sha256', 'insertion_order_sha256', 'live_ids_sha256', 'training', 'list_sizes'):
            if a[key] != b[key]:
                raise ValueError('same-training rebuild control differs')
        for regime in a['runs']:
            if a['runs'][regime]['sha256'] != b['runs'][regime]['sha256']:
                raise ValueError('same-training rebuild rankings differ')
    if first['snapshots'][0]['serialized_index_sha256'] != reports['refreshed_rebuild']['snapshots'][0]['serialized_index_sha256']:
        raise ValueError('snapshot-1 retrained baseline differs')
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'verification.json':
            if path.is_symlink() or path.stat().st_size > 32 * 1024**2:
                raise ValueError('unexpected unbounded IVF evidence file')
            artifacts[path.relative_to(output).as_posix()] = checksum(path)
    return {'status': 'verified', 'seed': first['seed'], 'same_training_rebuild_equal': True,
            'snapshot1_all_modes_equal': True, 'rows': rows, 'artifacts': artifacts,
            'limits': ['Full native indexes not exported; equality uses recorded serialization hashes',
                       'Canonical score checks were executed on the compute node; recheck locally against audited vectors after retrieval',
                       'No held-out or relevance-effectiveness claim']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('references', type=Path)
    parser.add_argument('query_contract', type=Path)
    args = parser.parse_args()
    roles = json.loads(args.query_contract.read_text())['roles']
    report = verify_experiment(args.output, args.references, roles)
    atomic_json(args.output / 'verification.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'seed', 'same_training_rebuild_equal', 'snapshot1_all_modes_equal')}))
