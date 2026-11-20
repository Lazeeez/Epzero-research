"""One-process, one-index HNSW development trajectories over audited mapped snapshots."""
from __future__ import annotations

import argparse
import gc
import json
import mmap
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .backends import Backend
from .contracts import digest
from .dense import ARTIFACTS
from .evaluation import ann_recall, read_run, validate_run, write_run
from .mapped import MappedSnapshot
from .query_diagnostic import load_queries, validate_inputs
from .release import checksum
from .runner import calibrate, measured, provenance
from .storage import atomic_json


def validate_settings(config, kind, mode, seed):
    required = {'schema', 'M', 'ef_construction', 'k', 'ef', 'ef_grid', 'recall_target', 'threads', 'batch_rows'}
    if set(config) != required or config['schema'] != 1 or config['threads'] != 1:
        raise ValueError('invalid trajectory configuration')
    if kind not in ('epzero', 'faiss_hnsw') or mode not in ('incremental', 'replay', 'shuffled'):
        raise ValueError('unsupported trajectory backend or mode')
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('invalid trajectory seed')
    for key in ('M', 'ef_construction', 'k', 'ef', 'batch_rows'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError('invalid positive integer setting')
    grid = config['ef_grid']
    if (config['M'] < 2 or config['ef'] < config['k'] or not grid
            or any(type(x) is not int or x < config['k'] for x in grid)
            or grid != sorted(set(grid)) or not 0 < config['recall_target'] <= 1):
        raise ValueError('invalid search grid or recall target')


def insert_mapped(backend, snapshot, order, batch_rows):
    """One ID map and a bounded vector copy, releasing touched mmap pages each batch."""
    if type(batch_rows) is not int or batch_rows <= 0:
        raise ValueError('invalid insertion batch size')
    if len(order) != len(set(order)) or backend.live.intersection(order):
        raise ValueError('duplicate/replacement insertion IDs')
    positions = {key: i for i, key in enumerate(snapshot.ids)}
    if set(order) - positions.keys() or set(order) - backend.labels.keys():
        raise ValueError('unknown insertion ID')
    if backend.kind not in ('epzero', 'faiss_hnsw'):
        raise ValueError('mapped insertion supports uncompressed HNSW only')
    for start in range(0, len(order), batch_rows):
        chunk = order[start:start+batch_rows]
        values = np.ascontiguousarray(snapshot.vectors[[positions[key] for key in chunk]])
        if backend.kind == 'epzero':
            labels = np.array([backend.labels[key] for key in chunk], dtype=np.int64)
            backend.index.add(values, labels=labels, threads=1)
        else:
            # Preserve the existing independent adapter's explicit one-row insertion order.
            for row in values:
                backend.index.add(row.reshape(1, -1))
        backend.live.update(chunk)
        backend.inserted.extend(chunk)
        del values
        if isinstance(snapshot.vectors, np.memmap) and hasattr(snapshot.vectors._mmap, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
            snapshot.vectors._mmap.madvise(mmap.MADV_DONTNEED)


def trajectory_core(specs, queries, references, output, config, *, kind, mode, seed, provenance_record,
                    snapshot_observer=None, evaluation_role='development'):
    """Internal executor; the public entry points validate real data/query audit chains.

    evaluation_role names the searched cohort: 'development' (training queries) or
    'evaluation' (held-out test queries). Calibration always uses the calibration role.
    """
    validate_settings(config, kind, mode, seed)
    if evaluation_role not in ('development', 'evaluation'):
        raise ValueError('unknown evaluation role')
    role = evaluation_role
    if [x['snapshot'] for x in specs] != [1, 2, 3] or len(references) != 3:
        raise ValueError('requires three ordered snapshots and references')
    if set(queries) != {role, 'calibration'} or set(queries[role].ids) & set(queries['calibration'].ids):
        raise ValueError('query roles must be separate')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    held_out = role == 'evaluation'
    report = {'status': 'running', 'purpose': ('held-out ANN trajectory under FINAL_PROTOCOL_V1' if held_out else
                                               'full-data development ANN trajectory; not primary research results'),
              'backend': kind, 'mode': mode, 'seed': seed, 'config': config, 'metric': 'ip',
              'provenance': provenance_record, 'snapshots': [], 'costs': [],
              'order_rule': 'lexical additions per snapshot; shuffled rebuild uses numpy default_rng(seed+snapshot-1)',
              'capacity_rule': 'Epzero reserves final snapshot count; FAISS retains its native growth policy',
              'memory_scope': 'sampled whole-worker RSS, one native index plus ID maps and insertion batch',
              'limits': [('Held-out test queries; no relevance judgments loaded' if held_out else
                          'Training queries only; no relevance judgments loaded'),
                         'Instrumentation overhead not yet characterized',
                         'Not isolated publication-quality latency evidence', 'IVF/matched-cost/primary inference not implemented here']}
    if held_out:
        report['evaluation_role'] = role
    atomic_json(output / 'report.json', report)
    backend = None
    try:
        final = specs[-1]
        with MappedSnapshot(final['folder'], expected_archive_sha256=final['archive_sha256']) as last:
            labels = tuple(sorted(last.ids))
            dimension = last.vectors.shape[1]
        report['label_map'] = {'rule': 'lexically sorted final snapshot external IDs', 'count': len(labels), 'sha256': digest(labels)}
        prior_ids, chronological, parent = set(), [], None
        for spec, reference in zip(specs, references):
            step = spec['snapshot']
            context = {'snapshot': step, 'backend': kind, 'mode': mode, 'seed': seed}
            measure = lambda name, action: measured(report['costs'], name, context, action)
            print(json.dumps({**context, 'stage': 'verify_and_build'}), flush=True)
            if mode != 'incremental' and backend is not None:
                del backend
                backend = None
                gc.collect()
            folder = Path(spec['folder'])
            if measure('verify_vector_bytes', lambda: checksum(folder / 'vecs.f4')) != spec['vector_sha256']:
                raise ValueError('snapshot vector bytes changed')
            with MappedSnapshot(folder, expected_archive_sha256=spec['archive_sha256']) as mapped:
                if (mapped.receipt['members']['vecs.f4']['sha256'] != spec['vector_sha256']
                        or mapped.vectors.shape[1] != dimension or len(mapped.ids) < config['k']):
                    raise ValueError('snapshot identity, dimension or size mismatch')
                current = set(mapped.ids)
                if prior_ids - current:
                    raise ValueError('insert-only snapshot removed IDs')
                additions = sorted(current - prior_ids)
                chronological.extend(additions)
                expected_query_ids = list(queries[role].ids) + list(queries['calibration'].ids)
                validate_run(reference, expected_query_ids, mapped.ids, config['k'])
                if backend is None:
                    backend = measure('allocate', lambda: Backend(kind, config, {'metric': 'ip', 'dim': dimension}, labels, seed))
                    order = list(chronological)
                    if mode == 'shuffled':
                        order = np.random.default_rng(seed + step - 1).permutation(order).tolist()
                else:
                    order = additions
                measure('insert_delta' if mode == 'incremental' and step > 1 else 'build',
                        lambda: insert_mapped(backend, mapped, order, config['batch_rows']))
            # The snapshot mmap is closed before calibration/search; the backend owns its vectors.
            # FAISS serializes efSearch. Previous-stage calibration must not alter the
            # construction comparison with a fresh replay index.
            if kind == 'faiss_hnsw':
                backend.index.hnsw.efSearch = config['ef']
            state = measure('fingerprint', backend.state)
            if kind == 'faiss_hnsw':
                state['fingerprint_ef_search'] = config['ef']
            if state['live_set_sha256'] != digest(sorted(current)):
                raise ValueError('index live-set mismatch')
            state.update(snapshot=step, parent_state_sha256=parent if mode == 'incremental' else None,
                         snapshot_archive_sha256=spec['archive_sha256'])
            state['state_sha256'] = digest(state)
            parent = state['state_sha256']
            cal_ref = {q: reference[q] for q in queries['calibration'].ids}
            selected = measure('calibration', lambda: calibrate(backend, queries['calibration'], cal_ref,
                               config['k'], config['ef_grid'], config['recall_target']))
            stage = {'snapshot': step, 'added_count': len(additions), 'state': state,
                     'calibration': selected, 'runs': {}}
            dev_ref = {q: reference[q] for q in queries[role].ids}
            for regime, effort in (('frozen', config['ef']), ('matched_recall', selected['effort'])):
                measure(regime + '_warmup', lambda: backend.search(queries[role], config['k'], effort))
                run = measure(regime + '_search', lambda: backend.search(queries[role], config['k'], effort))
                name = f'snapshot-{step}.{regime}.run'
                write_run(output / name, run)
                reread = read_run(output / name)
                if run != reread:
                    raise ValueError('run export roundtrip changed rankings or scores')
                recalls = ann_recall(run, dev_ref)
                stage['runs'][regime] = {'effort': effort, 'file': name, 'sha256': checksum(output / name),
                    f'{role}_query_count': len(recalls), 'ann_recall_mean': float(np.mean(list(recalls.values()))),
                    'ann_recall_per_query': recalls}
            report['snapshots'].append(stage)
            atomic_json(output / 'report.json', report)
            if snapshot_observer is not None:
                stage['diagnostic'] = measure('diagnostic_observer', lambda: snapshot_observer(
                    backend, stage, queries, reference, config))
                atomic_json(output / 'report.json', report)
            print(json.dumps({**context, 'stage': 'completed', 'live_count': len(current),
                              'calibration_target_met': selected['target_met']}), flush=True)
            prior_ids = current
        report['status'] = 'completed'
        atomic_json(output / 'report.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__)
        atomic_json(output / 'report.json', report)
        raise
    finally:
        del backend
        gc.collect()


def run_trajectory(folder, audit_path, text_audit_path, inputs, encoded, references, output, config_path,
                   *, kind='epzero', mode='incremental', seed=11, native_origin='unverified', snapshot_observer=None):
    folder, audit_path, text_audit_path, inputs, encoded, references, output, config_path = map(
        Path, (folder, audit_path, text_audit_path, inputs, encoded, references, output, config_path))
    config = json.loads(config_path.read_text())
    validate_settings(config, kind, mode, seed)
    if config['k'] != 100:
        raise ValueError('real development reference contract requires depth 100')
    if any(os.environ.get(key) != '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('set all BLAS/OpenMP thread environment variables to 1')
    audit, text = json.loads(audit_path.read_text()), json.loads(text_audit_path.read_text())
    if (audit.get('status') != 'completed' or audit.get('fixed_vector_gate_passed') is not True
            or text.get('status') != 'completed' or text.get('fixed_text_gate_passed') is not True
            or [x['snapshot'] for x in audit['snapshots']] != [1, 2, 3]
            or [x['snapshot'] for x in text['snapshots']] != [1, 2, 3]):
        raise ValueError('requires full fixed-vector and fixed-text audits')
    ids, values, raw = load_queries(inputs, encoded)
    if raw['dtype'] != 'float32' or raw['device'] != 'cuda' or raw.get('tf32') is not False:
        raise ValueError('development trajectory uses the declared CUDA float32 candidate only')
    contract, _ = validate_inputs(inputs)
    positions = {qid: row for row, qid in enumerate(ids)}
    queries = {role: SimpleNamespace(ids=keys, vectors=np.ascontiguousarray(values[[positions[q] for q in keys]]))
               for role, keys in contract['roles'].items()}
    specs, refs, evidence = [], [], {}
    for vector, texts, artifact in zip(audit['snapshots'], text['snapshots'], ARTIFACTS):
        step = artifact['snapshot']
        if (vector['archive_sha256'] != artifact['archive_sha256'] or not vector['scan']['numeric_vectors_valid']
                or texts['conflicting_duplicate_rows'] or texts['dense_coverage']['missing_text_ids']
                or texts['dense_coverage']['unindexed_text_ids']
                or texts['dense_coverage']['id_map_sha256'] != vector['members']['docnos.npids']['sha256']):
            raise ValueError('data audit identity or coverage mismatch')
        path = references / f'snapshot-{step}'
        receipt = json.loads((path / 'report.json').read_text())
        run_path = path / 'float32.run'
        if (receipt.get('status') != 'completed' or receipt['snapshot'] != step or receipt['depth'] != config['k']
                or receipt['vector_sha256'] != vector['scan']['vector_sha256']
                or receipt['fp32_receipt_sha256'] != checksum(encoded / 'query-encoding.json')
                or receipt['query_inputs_sha256'] != checksum(inputs / 'query-inputs.json')
                or checksum(run_path) != receipt['artifacts']['float32.run']):
            raise ValueError('exact reference provenance mismatch')
        specs.append({'snapshot': step, 'folder': folder / f'snapshot-{step}',
                      'archive_sha256': artifact['archive_sha256'], 'vector_sha256': vector['scan']['vector_sha256']})
        refs.append(read_run(run_path))
        evidence[f'snapshot-{step}'] = {'report_sha256': checksum(path / 'report.json'), 'run_sha256': checksum(run_path)}
    record = provenance(native_origin)
    record['development_inputs'] = {'vector_audit_sha256': checksum(audit_path), 'text_audit_sha256': checksum(text_audit_path),
        'query_inputs_sha256': checksum(inputs / 'query-inputs.json'), 'query_encoding_sha256': checksum(encoded / 'query-encoding.json'),
        'config_sha256': checksum(config_path), 'exact_references': evidence}
    return trajectory_core(specs, queries, refs, output, config, kind=kind, mode=mode, seed=seed,
                           provenance_record=record, snapshot_observer=snapshot_observer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('folder', 'audit', 'text_audit', 'inputs', 'encoded', 'references', 'output'): parser.add_argument(name, type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('trajectory-development.json'))
    parser.add_argument('--backend', choices=('epzero', 'faiss_hnsw'), default='epzero')
    parser.add_argument('--mode', choices=('incremental', 'replay', 'shuffled'), default='incremental')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--native-origin', choices=('unverified', 'checkout-build', 'released-wheel'), default='unverified')
    args = parser.parse_args()
    result = run_trajectory(args.folder, args.audit, args.text_audit, args.inputs, args.encoded, args.references,
        args.output, args.config, kind=args.backend, mode=args.mode, seed=args.seed, native_origin=args.native_origin)
    print(json.dumps({'status': result['status'], 'snapshots': len(result['snapshots']), 'backend': result['backend'], 'mode': result['mode']}))
