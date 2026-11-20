"""Held-out (test-query) execution for FINAL_PROTOCOL_V1: exact references and ANN arms.

Nothing here reads relevance judgments. Calibration always uses the 50 training calibration
queries and their verified development exact references; the searched cohort is the 381
released test queries (role 'evaluation'). Inputs use one fixed folder layout:

    inputs/training/queries/      query-inputs.json, records.jsonl, ...
    inputs/training/float32/      job-7107422 query encoding (query-encoding.json, encoder/)
    inputs/training/exact/snapshot-N/   report.json, float32.run (development references)
    inputs/training/text-audit.json
    inputs/test/inputs/           test-inputs.json, test-records.jsonl, ...
    inputs/test/encoding/         job-7164678 encoding (test-encoding.json, test/, training/)
    inputs/test/exact/snapshot-N/ report.json, evaluation.run (held-out exact references)
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np

from .contracts import digest
from .dense import ARTIFACTS
from .evaluation import read_run, validate_run, write_run
from .mapped import MappedSnapshot
from .oracle import exact_mapped
from .query_diagnostic import load_queries, validate_inputs
from .release import checksum
from .storage import atomic_json
from .test_queries import load_test_queries

ROLE = 'evaluation'
DEPTH = 100
SOURCES = ('heldout.py', 'oracle.py', 'mapped.py', 'trajectory.py', 'ivf_trajectory.py', 'backends.py',
           'evaluation.py', 'runner.py', 'test_queries.py', 'query_diagnostic.py')


def _training_queries(inputs):
    training = Path(inputs) / 'training'
    ids, values, raw = load_queries(training / 'queries', training / 'float32')
    if raw['dtype'] != 'float32' or raw['device'] != 'cuda' or raw.get('tf32') is not False:
        raise ValueError('calibration queries must use the declared CUDA float32 representation')
    contract, _ = validate_inputs(training / 'queries')
    positions = {qid: row for row, qid in enumerate(ids)}
    keys = list(contract['roles']['calibration'])
    calibration = SimpleNamespace(ids=keys, vectors=np.ascontiguousarray(values[[positions[q] for q in keys]]))
    return calibration, set(ids)


def _test_queries(inputs, training_ids):
    test = Path(inputs) / 'test'
    ids, values = load_test_queries(test / 'inputs', test / 'encoding')
    report = json.loads((test / 'encoding/test-encoding.json').read_text())
    if report['dtype'] != 'float32' or report['device'] != 'cuda' or report.get('tf32') is not False:
        raise ValueError('test queries must use the declared CUDA float32 representation')
    if set(ids) & training_ids:
        raise ValueError('test and training query IDs overlap')
    return SimpleNamespace(ids=list(ids), vectors=np.ascontiguousarray(values))


def _development_reference(inputs, step):
    training = Path(inputs) / 'training'
    folder = training / 'exact' / f'snapshot-{step}'
    receipt = json.loads((folder / 'report.json').read_text())
    if (receipt.get('status') != 'completed' or receipt['snapshot'] != step or receipt['depth'] != DEPTH
            or receipt['query_inputs_sha256'] != checksum(training / 'queries/query-inputs.json')
            or receipt['fp32_receipt_sha256'] != checksum(training / 'float32/query-encoding.json')
            or receipt['artifacts']['float32.run'] != checksum(folder / 'float32.run')):
        raise ValueError('development exact reference identity mismatch')
    return receipt, read_run(folder / 'float32.run')


def _test_reference(inputs, step, test_ids):
    test = Path(inputs) / 'test'
    folder = test / 'exact' / f'snapshot-{step}'
    receipt = json.loads((folder / 'report.json').read_text())
    if (receipt.get('status') != 'completed' or receipt['snapshot'] != step or receipt['depth'] != DEPTH
            or receipt['test_inputs_sha256'] != checksum(test / 'inputs/test-inputs.json')
            or receipt['test_encoding_sha256'] != checksum(test / 'encoding/test-encoding.json')
            or receipt['artifacts']['evaluation.run'] != checksum(folder / 'evaluation.run')):
        raise ValueError('held-out exact reference identity mismatch')
    run = read_run(folder / 'evaluation.run')
    if set(run) != set(test_ids):
        raise ValueError('held-out exact reference population mismatch')
    return receipt, run


def load_heldout(inputs, *, references=True):
    """Validated calibration/evaluation cohorts and, optionally, combined exact references."""
    inputs = Path(inputs)
    calibration, training_ids = _training_queries(inputs)
    evaluation = _test_queries(inputs, training_ids)
    queries = {'calibration': calibration, ROLE: evaluation}
    evidence = {'training_query_inputs_sha256': checksum(inputs / 'training/queries/query-inputs.json'),
                'training_query_encoding_sha256': checksum(inputs / 'training/float32/query-encoding.json'),
                'test_inputs_sha256': checksum(inputs / 'test/inputs/test-inputs.json'),
                'test_encoding_sha256': checksum(inputs / 'test/encoding/test-encoding.json'),
                'calibration_count': len(calibration.ids), 'evaluation_count': len(evaluation.ids)}
    if not references:
        return queries, None, None, evidence
    refs, receipts = [], []
    for step in (1, 2, 3):
        development, dev_run = _development_reference(inputs, step)
        test, test_run = _test_reference(inputs, step, evaluation.ids)
        if development['vector_sha256'] != test['vector_sha256']:
            raise ValueError('exact references were computed on different snapshot vectors')
        refs.append({**{q: dev_run[q] for q in calibration.ids}, **test_run})
        receipts.append({'snapshot': step, 'vector_sha256': test['vector_sha256'],
                         'development_report_sha256': checksum(inputs / f'training/exact/snapshot-{step}/report.json'),
                         'heldout_report_sha256': checksum(inputs / f'test/exact/snapshot-{step}/report.json')})
    evidence['exact_references'] = receipts
    return queries, refs, receipts, evidence


def exact_evaluation(folder, scan_path, inputs, output, *, snapshot, block_rows=2048):
    """Float64 exhaustive top-100 for the test queries at one snapshot.

    The calibration queries are re-scored in a separate call and compared with the verified
    development references as an implementation check; those references remain authoritative.
    """
    folder, scan_path, inputs, output = map(Path, (folder, scan_path, inputs, output))
    if snapshot not in (1, 2, 3) or type(block_rows) is not int or block_rows <= 0:
        raise ValueError('invalid snapshot or block size')
    queries, _, _, evidence = load_heldout(inputs, references=False)
    dev_receipt, dev_run = _development_reference(inputs, snapshot)
    scan = json.loads(scan_path.read_text())
    if scan.get('status') != 'completed' or not scan.get('numeric_vectors_valid'):
        raise ValueError('requires a completed valid vector audit')
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'purpose': 'held-out exact references under FINAL_PROTOCOL_V1; no judgments loaded',
              'snapshot': snapshot, 'depth': DEPTH, 'metric': 'ip', 'query_count': len(queries[ROLE].ids),
              'accumulation': 'float64 over original float32 inputs',
              'tie_rule': 'score descending then external ID descending', 'block_rows': block_rows,
              'source_sha256': {name: checksum(Path(__file__).with_name(name)) for name in SOURCES},
              'test_inputs_sha256': evidence['test_inputs_sha256'],
              'test_encoding_sha256': evidence['test_encoding_sha256'], 'scan_sha256': checksum(scan_path),
              'development_reference_sha256': checksum(inputs / f'training/exact/snapshot-{snapshot}/report.json')}
    atomic_json(output / 'report.json', report)
    try:
        with MappedSnapshot(folder, expected_archive_sha256=ARTIFACTS[snapshot - 1]['archive_sha256']) as mapped:
            if (scan['vector_sha256'] != mapped.receipt['members']['vecs.f4']['sha256']
                    or checksum(folder / 'vecs.f4') != scan['vector_sha256']
                    or dev_receipt['vector_sha256'] != scan['vector_sha256']):
                raise ValueError('reference vectors differ from audit, archive or development reference')
            report['vector_sha256'] = scan['vector_sha256']
            started = time.perf_counter()
            run = exact_mapped(mapped.ids, mapped.vectors, queries[ROLE].ids, queries[ROLE].vectors, k=DEPTH,
                               metric='ip', block_rows=block_rows,
                               progress=lambda value: atomic_json(output / 'progress.json', value))
            report['exact_seconds'] = time.perf_counter() - started
            validate_run(run, queries[ROLE].ids, mapped.ids, DEPTH)
            write_run(output / 'evaluation.run', run)
            if read_run(output / 'evaluation.run') != run:
                raise ValueError('exact run export roundtrip changed rankings or scores')
            check = exact_mapped(mapped.ids, mapped.vectors, queries['calibration'].ids, queries['calibration'].vectors,
                                 k=DEPTH, metric='ip', block_rows=block_rows)
        report['calibration_recheck'] = reference_agreement(check, {q: dev_run[q] for q in queries['calibration'].ids})
        if not report['calibration_recheck']['passed']:
            raise ValueError('calibration re-scoring disagrees with the development reference')
        report['artifacts'] = {'evaluation.run': checksum(output / 'evaluation.run')}
        report['status'] = 'completed'
        atomic_json(output / 'report.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__)
        atomic_json(output / 'report.json', report)
        raise


def reference_agreement(left, right, *, atol=1e-9, min_overlap=.98):
    """Two exhaustive runs may differ only by float rounding near ties."""
    if set(left) != set(right):
        raise ValueError('reference populations differ')
    overlaps, errors, identical = [], [], 0
    for q in left:
        a, b = dict(left[q]), dict(right[q])
        overlaps.append(len(a.keys() & b.keys()) / len(b))
        errors.extend(abs(a[d] - b[d]) for d in a.keys() & b.keys())
        identical += left[q] == right[q]
    return {'queries': len(left), 'identical_rankings': identical, 'min_overlap': min(overlaps),
            'mean_overlap': float(np.mean(overlaps)), 'max_abs_score_difference': max(errors),
            'atol': atol, 'min_overlap_required': min_overlap,
            'passed': min(overlaps) >= min_overlap and max(errors) <= atol}


def heldout_provenance(package_root, native_origin):
    """Package-based provenance: no git checkout exists on the compute node."""
    import epzero
    import epzero._core
    package_root = Path(package_root)
    sums = package_root / 'SHA256SUMS'
    return {'package_sha256sums_sha256': checksum(sums) if sums.exists() else None,
            'package_manifest_sha256': checksum(package_root / 'manifest.json') if (package_root / 'manifest.json').exists() else None,
            'research_sources': {name: checksum(Path(__file__).with_name(name)) for name in SOURCES},
            'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
            'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'faiss-cpu', 'epzero', 'psutil')},
            'native_origin_declared': native_origin, 'epzero_binary_sha256': checksum(Path(epzero._core.__file__)),
            'epzero_machine': epzero.machine(),
            'thread_environment': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}


def snapshot_specs(folder, audit_path, inputs, receipts):
    """Data-audit chain shared by both ANN arms; mirrors trajectory.run_trajectory."""
    folder, audit_path, inputs = Path(folder), Path(audit_path), Path(inputs)
    audit = json.loads(audit_path.read_text())
    text = json.loads((inputs / 'training/text-audit.json').read_text())
    if (audit.get('status') != 'completed' or audit.get('fixed_vector_gate_passed') is not True
            or text.get('status') != 'completed' or text.get('fixed_text_gate_passed') is not True
            or [x['snapshot'] for x in audit['snapshots']] != [1, 2, 3]
            or [x['snapshot'] for x in text['snapshots']] != [1, 2, 3]):
        raise ValueError('requires full fixed-vector and fixed-text audits')
    specs = []
    for vector, texts, artifact, receipt in zip(audit['snapshots'], text['snapshots'], ARTIFACTS, receipts):
        if (vector['archive_sha256'] != artifact['archive_sha256'] or not vector['scan']['numeric_vectors_valid']
                or texts['conflicting_duplicate_rows'] or texts['dense_coverage']['missing_text_ids']
                or texts['dense_coverage']['unindexed_text_ids']
                or texts['dense_coverage']['id_map_sha256'] != vector['members']['docnos.npids']['sha256']
                or receipt['vector_sha256'] != vector['scan']['vector_sha256']):
            raise ValueError('data audit or exact reference identity mismatch')
        specs.append({'snapshot': artifact['snapshot'], 'folder': folder / f"snapshot-{artifact['snapshot']}",
                      'archive_sha256': artifact['archive_sha256'], 'vector_sha256': vector['scan']['vector_sha256']})
    return specs, {'vector_audit_sha256': checksum(audit_path), 'text_audit_sha256': checksum(inputs / 'training/text-audit.json')}


def _require_single_thread():
    if any(os.environ.get(key) != '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('set all BLAS/OpenMP thread environment variables to 1')


def check(inputs, hnsw_config, ivf_config):
    """Preflight before downloads: inputs, references and both configurations."""
    from .ivf_trajectory import validate_settings as ivf_settings
    from .trajectory import validate_settings as hnsw_settings
    queries, refs, receipts, evidence = load_heldout(inputs)
    hnsw, ivf = json.loads(Path(hnsw_config).read_text()), json.loads(Path(ivf_config).read_text())
    hnsw_settings(hnsw, 'epzero', 'incremental', 11)
    ivf_settings(ivf, 'frozen_incremental', 11)
    if hnsw['k'] != DEPTH or ivf['k'] != DEPTH:
        raise ValueError('held-out references require depth 100')
    for reference in refs:
        if set(reference) != set(queries['calibration'].ids) | set(queries[ROLE].ids):
            raise ValueError('combined reference population mismatch')
    return {'status': 'preflight_passed', **evidence,
            'hnsw_config_sha256': checksum(hnsw_config), 'ivf_config_sha256': checksum(ivf_config)}


def run_hnsw(folder, audit_path, inputs, output, config_path, *, kind, mode, seed, package_root, native_origin):
    from .trajectory import trajectory_core, validate_settings
    config = json.loads(Path(config_path).read_text())
    validate_settings(config, kind, mode, seed)
    if mode not in ('incremental', 'replay') or config['k'] != DEPTH:
        raise ValueError('held-out HNSW arms are incremental and identical-history replay at depth 100')
    _require_single_thread()
    queries, refs, receipts, evidence = load_heldout(inputs)
    specs, audits = snapshot_specs(folder, audit_path, inputs, receipts)
    record = heldout_provenance(package_root, native_origin)
    record['heldout_inputs'] = {**evidence, **audits, 'config_sha256': checksum(config_path)}
    return trajectory_core(specs, queries, refs, output, config, kind=kind, mode=mode, seed=seed,
                           provenance_record=record, evaluation_role=ROLE)


def run_ivf(folder, audit_path, inputs, output, config_path, *, mode, seed, package_root, native_origin):
    from .ivf_trajectory import ivf_core, validate_settings
    config = json.loads(Path(config_path).read_text())
    validate_settings(config, mode, seed)
    if config['k'] != DEPTH:
        raise ValueError('held-out IVF arm requires depth 100')
    _require_single_thread()
    queries, refs, receipts, evidence = load_heldout(inputs)
    specs, audits = snapshot_specs(folder, audit_path, inputs, receipts)
    record = heldout_provenance(package_root, native_origin)
    record['heldout_inputs'] = {**evidence, **audits, 'config_sha256': checksum(config_path)}
    return ivf_core(specs, queries, refs, output, config, mode=mode, seed=seed,
                    provenance_record=record, evaluation_role=ROLE)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('exact')
    for name in ('folder', 'scan', 'inputs', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--snapshot', type=int, required=True)
    p.add_argument('--block-rows', type=int, default=2048)
    p = sub.add_parser('check')
    for name in ('inputs', 'hnsw_config', 'ivf_config'):
        p.add_argument(name, type=Path)
    for command in ('hnsw', 'ivf'):
        p = sub.add_parser(command)
        for name in ('folder', 'audit', 'inputs', 'output', 'config'):
            p.add_argument(name, type=Path)
        if command == 'hnsw':
            p.add_argument('--backend', choices=('epzero', 'faiss_hnsw'), required=True)
            p.add_argument('--mode', choices=('incremental', 'replay'), required=True)
        else:
            p.add_argument('--mode', choices=('frozen_incremental', 'frozen_rebuild', 'refreshed_rebuild'), required=True)
        p.add_argument('--seed', type=int, choices=(11, 29, 47), required=True)
        p.add_argument('--package-root', type=Path, required=True)
        p.add_argument('--native-origin', choices=('unverified', 'checkout-build', 'released-wheel'), default='unverified')
    a = parser.parse_args()
    if a.command == 'exact':
        r = exact_evaluation(a.folder, a.scan, a.inputs, a.output, snapshot=a.snapshot, block_rows=a.block_rows)
        r = {k: r[k] for k in ('status', 'snapshot', 'query_count', 'exact_seconds', 'calibration_recheck')}
    elif a.command == 'check':
        r = check(a.inputs, a.hnsw_config, a.ivf_config)
    elif a.command == 'hnsw':
        r = run_hnsw(a.folder, a.audit, a.inputs, a.output, a.config, kind=a.backend, mode=a.mode, seed=a.seed,
                     package_root=a.package_root, native_origin=a.native_origin)
        r = {k: r[k] for k in ('status', 'backend', 'mode', 'seed')}
    else:
        r = run_ivf(a.folder, a.audit, a.inputs, a.output, a.config, mode=a.mode, seed=a.seed,
                    package_root=a.package_root, native_origin=a.native_origin)
        r = {k: r[k] for k in ('status', 'mode', 'seed')}
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
