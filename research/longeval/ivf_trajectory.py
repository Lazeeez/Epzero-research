"""Bounded full-data IVFFlat development controls; no judgments or held-out queries."""
from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import mmap
import os
from pathlib import Path
import time
from types import SimpleNamespace

import faiss
import numpy as np

from .backends import streamed_faiss_hash
from .contracts import digest
from .dense import ARTIFACTS
from .evaluation import ann_recall, rank, read_run, validate_run, write_run
from .mapped import MappedSnapshot
from .query_diagnostic import load_queries, validate_inputs
from .release import checksum
from .runner import measured, provenance
from .storage import atomic_json


MODES = ('frozen_incremental', 'frozen_rebuild', 'refreshed_rebuild')


def validate_settings(config, mode, seed):
    required = {'schema', 'nlist', 'training_rows', 'iterations', 'k', 'nprobe_grid',
                'recall_target', 'threads', 'batch_rows', 'search_repeats'}
    if set(config) != required or config['schema'] != 1 or config['threads'] != 1:
        raise ValueError('invalid IVF configuration')
    if mode not in MODES or type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('invalid mode or seed')
    for key in ('nlist', 'training_rows', 'iterations', 'k', 'batch_rows', 'search_repeats'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError('invalid positive integer setting')
    grid = config['nprobe_grid']
    if (not grid or any(type(p) is not int or not 1 <= p <= config['nlist'] for p in grid)
            or grid != sorted(set(grid)) or grid[-1] != config['nlist']
            or config['training_rows'] < config['nlist'] or not 0 < config['recall_target'] <= 1):
        raise ValueError('invalid training sample, probe grid, or target')


def sample_ids(ids, count, seed):
    """Bottom SHA-256 priorities: row-order independent, bounded heap, nested population."""
    if count <= 0 or len(ids) < count:
        raise ValueError('training sample exceeds population')
    prefix = f'longeval-ivf-training-v1:{seed}:'.encode()
    selected = heapq.nsmallest(count, ids, key=lambda key: (hashlib.sha256(prefix + key.encode()).digest(), key))
    return sorted(selected)


def release_pages(values):
    if isinstance(values, np.memmap) and hasattr(values._mmap, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
        values._mmap.madvise(mmap.MADV_DONTNEED)


def gather(snapshot, ids, batch_rows):
    positions = {key: i for i, key in enumerate(snapshot.ids)}
    result = np.empty((len(ids), snapshot.vectors.shape[1]), dtype='float32')
    for start in range(0, len(ids), batch_rows):
        chunk = ids[start:start+batch_rows]
        result[start:start+len(chunk)] = snapshot.vectors[[positions[q] for q in chunk]]
        release_pages(snapshot.vectors)
    return result


def train_template(snapshot, config, seed, destination):
    ids = sample_ids(snapshot.ids, config['training_rows'], seed)
    values = gather(snapshot, ids, config['batch_rows'])
    dim = values.shape[1]
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, config['nlist'], faiss.METRIC_INNER_PRODUCT)
    index.cp.seed = seed
    index.cp.niter = config['iterations']
    index.cp.nredo = 1
    index.cp.spherical = True
    # Prevent FAISS from subsampling our explicitly selected training matrix.
    index.cp.max_points_per_centroid = max(256, (len(ids) + config['nlist'] - 1) // config['nlist'])
    index.train(values)
    if not index.is_trained or index.ntotal or index.quantizer.ntotal != config['nlist']:
        raise ValueError('unexpected trained template state')
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / 'ids.json', ids)
    centroids = index.quantizer.reconstruct_n(0, config['nlist'])
    np.save(destination / 'centroids.npy', centroids, allow_pickle=False)
    result = {'count': len(ids), 'sample_rule': 'bottom SHA256(longeval-ivf-training-v1:seed:ID), lexical training order',
              'sample_ids_sha256': digest(ids), 'training_vectors_sha256': hashlib.sha256(values).hexdigest(),
              'centroids_sha256': hashlib.sha256(centroids).hexdigest(),
              'empty_index_sha256': streamed_faiss_hash(index),
              'clustering': {'seed': seed, 'niter': config['iterations'], 'nredo': 1, 'spherical': True,
                             'max_points_per_centroid': index.cp.max_points_per_centroid},
              'files': {name: checksum(destination / name) for name in ('ids.json', 'centroids.npy')}}
    return index, result


def add_mapped(index, snapshot, order, labels, batch_rows):
    if len(set(order)) != len(order) or not set(order) <= set(snapshot.ids):
        raise ValueError('duplicate or absent insertion IDs')
    positions = {key: row for row, key in enumerate(snapshot.ids)}
    for start in range(0, len(order), batch_rows):
        chunk = order[start:start+batch_rows]
        values = np.ascontiguousarray(snapshot.vectors[[positions[key] for key in chunk]])
        index.add_with_ids(values, np.array([labels[key] for key in chunk], dtype=np.int64))
        del values
        release_pages(snapshot.vectors)


def search_cohort(index, ids, queries, k, nprobe, *, repeats=1):
    """Single-query native calls; one warmup then timed repeats, counters independently checked.

    ndis counts scanned full vectors, not coarse-centroid comparisons. The latter are
    nlist for this IndexFlatIP quantizer and are recorded separately. No max_codes cap.
    Python validation and explicit list-length calculation are outside the timed call.
    """
    if not 1 <= nprobe <= index.nlist or repeats < 1:
        raise ValueError('invalid probe count or repeats')
    index.nprobe = nprobe
    index.max_codes = 0
    run, records, incomplete = {}, [], []
    for qid, vector in zip(queries.ids, queries.vectors):
        query = np.ascontiguousarray(vector.reshape(1, -1))
        _, probed = index.quantizer.search(query, nprobe)
        expected = sum(index.invlists.list_size(int(p)) for p in probed[0])
        first = None
        times = []
        for repeat in range(repeats + 1):
            faiss.cvar.indexIVF_stats.reset()
            start = time.perf_counter_ns()
            scores, labels = index.search(query, k)
            elapsed = time.perf_counter_ns() - start
            work = int(faiss.cvar.indexIVF_stats.ndis)
            if work != expected:
                raise ValueError('native scan count differs from probed list lengths')
            if first is None:
                first = (scores.copy(), labels.copy())
            elif not np.array_equal(scores, first[0]) or not np.array_equal(labels, first[1]):
                raise ValueError('search repeat changed results')
            if repeat:
                times.append(elapsed)
        valid = labels[0] >= 0
        if int(valid.sum()) != k:
            incomplete.append(qid)
        if any(int(label) >= len(ids) for label in labels[0][valid]):
            raise ValueError('unknown native label')
        pairs = [(ids[int(label)], float(value)) for label, value in zip(labels[0][valid], scores[0][valid])]
        if len({d for d, _ in pairs}) != len(pairs) or not all(np.isfinite(s) for _, s in pairs):
            raise ValueError('duplicate ID or nonfinite score')
        run[qid] = rank(pairs)
        records.append({'query_id': qid, 'returned': len(pairs), 'scanned_vectors': work,
                        'probed_lists': [int(p) for p in probed[0]],
                        'coarse_centroid_comparisons': index.nlist, 'total_vector_comparisons': work + index.nlist,
                        'native_call_nanoseconds': times})
    return run, {'feasible': not incomplete, 'incomplete_queries': incomplete, 'nprobe': nprobe,
                 'warmups_per_query': 1, 'timed_repeats': repeats, 'queries': records}


def select_probe(points, target):
    valid = [p for p in points if p['feasible']]
    if not valid:
        raise ValueError('no feasible calibration probe count')
    meets = [p for p in valid if p['mean_recall'] >= target]
    chosen = min(meets, key=lambda p: p['nprobe']) if meets else max(valid, key=lambda p: (p['mean_recall'], -p['nprobe']))
    return {'nprobe': chosen['nprobe'], 'mean_calibration_recall': chosen['mean_recall'],
            'target': target, 'target_met': bool(meets)}


def export_run(path, run):
    """TREC has no row for an empty query; its ID remains in the point metadata."""
    write_run(path, run)
    if read_run(path) != {q: pairs for q, pairs in run.items() if pairs}:
        raise ValueError('run export changed results')


def verify_returned_scores(snapshot, queries, runs, references, *, batch_rows=1024, atol=1e-5):
    """Independent float64 scores; exhaustive arm may differ only near the boundary."""
    positions = {key: row for row, key in enumerate(snapshot.ids)}
    query_vectors = {q: v for table in queries.values() for q, v in zip(table.ids, table.vectors)}
    observed = {}
    for run, exhaustive in runs:
        for q, pairs in run.items():
            for doc, value in pairs:
                if doc not in positions:
                    raise ValueError('returned document is not live')
                observed.setdefault(q, {}).setdefault(doc, []).append(value)
    canonical, max_error, count = {}, 0., 0
    for q, pairs in observed.items():
        canonical[q] = {}
        docs = list(pairs)
        query = query_vectors[q].astype(np.float64)
        for start in range(0, len(docs), batch_rows):
            chunk = docs[start:start+batch_rows]
            vectors = np.array(snapshot.vectors[[positions[d] for d in chunk]], dtype=np.float64)
            scores = vectors @ query
            for doc, value in zip(chunk, scores):
                canonical[q][doc] = float(value)
                error = max(abs(float(value) - s) for s in pairs[doc])
                max_error = max(max_error, error)
                if error > atol:
                    raise ValueError('returned score differs from canonical float64 dot product')
            count += len(chunk)
            release_pages(snapshot.vectors)
    substitutions = 0
    for run, exhaustive in runs:
        if not exhaustive:
            continue
        for q, pairs in run.items():
            reference = references[q]
            if len(pairs) != len(reference):
                raise ValueError('exhaustive IVF returned a short ranking')
            cutoff = reference[-1][1]
            returned = {d for d, _ in pairs}
            missing = [(d, s) for d, s in reference if d not in returned]
            if any(s > cutoff + atol for _, s in missing) or any(canonical[q][d] < cutoff - atol for d in returned):
                raise ValueError('exhaustive IVF misses a non-boundary exact neighbor')
            substitutions += len(missing)
    return {'status': 'verified', 'distinct_query_document_pairs': count,
            'max_absolute_score_error': max_error, 'absolute_tolerance': atol,
            'all_list_boundary_substitutions': substitutions}


def verify_membership(index, ids, current):
    """Read the actual inverted-list labels, including duplicates and absent live IDs."""
    seen = np.zeros(len(ids), dtype=bool)
    for number in range(index.nlist):
        size = index.invlists.list_size(number)
        if not size:
            continue
        pointer = index.invlists.get_ids(number)
        try:
            labels = faiss.rev_swig_ptr(pointer, size)
            if np.any(labels < 0) or np.any(labels >= len(ids)):
                raise ValueError('invalid stored index label')
            if np.any(seen[labels]) or len(np.unique(labels)) != size:
                raise ValueError('duplicate stored index labels')
            seen[labels] = True
        finally:
            index.invlists.release_ids(number, pointer)
    actual = [ids[i] for i in np.flatnonzero(seen)]
    if set(actual) != current or len(actual) != index.ntotal:
        raise ValueError('stored index membership mismatch')
    return digest(actual)


def ivf_core(specs, queries, references, output, config, *, mode, seed, provenance_record,
             evaluation_role='development'):
    """evaluation_role is 'development' (training queries) or 'evaluation' (held-out test queries)."""
    validate_settings(config, mode, seed)
    if evaluation_role not in ('development', 'evaluation'):
        raise ValueError('unknown evaluation role')
    role = evaluation_role
    faiss.omp_set_num_threads(1)
    if [s['snapshot'] for s in specs] != [1, 2, 3] or len(references) != 3:
        raise ValueError('requires three ordered snapshots')
    if set(queries) != {'calibration', role} or set(queries['calibration'].ids) & set(queries[role].ids):
        raise ValueError('separate evaluation and calibration roles required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'backend': 'faiss_ivf', 'mode': mode, 'seed': seed,
              'purpose': ('held-out IVF arm under FINAL_PROTOCOL_V1' if role == 'evaluation' else
                          'development IVF maintenance controls; not held-out results'), 'config': config,
              'metric': 'inner_product', 'compression': 'none', 'provenance': provenance_record,
              'costs': [], 'snapshots': [],
              'limits': [('Held-out test queries; no relevance labels' if role == 'evaluation' else
                          'Training queries only; no relevance labels'),
                         'Native latency includes Python/native boundary and warm cache; not service throughput',
                         'Stage RSS includes the full worker; not isolated index allocation',
                         'IVF scan counts exclude coarse work; total comparisons include both',
                         'Fixed nprobe is not fixed search work as list sizes grow']}
    if role == 'evaluation':
        report['evaluation_role'] = role
    atomic_json(output / 'report.json', report)
    index = template = None
    training = None
    try:
        with MappedSnapshot(specs[-1]['folder'], expected_archive_sha256=specs[-1]['archive_sha256']) as last:
            ids = tuple(sorted(last.ids))
            dim = last.vectors.shape[1]
        labels = {key: row for row, key in enumerate(ids)}
        report['labels_sha256'] = digest(ids)
        previous, chronological, frozen_probe = set(), [], None
        for spec, reference in zip(specs, references):
            step = spec['snapshot']
            identity = {'snapshot': step, 'mode': mode, 'seed': seed}
            measure = lambda name, action: measured(report['costs'], name, identity, action)
            print(json.dumps({**identity, 'stage': 'build'}), flush=True)
            if mode != 'frozen_incremental':
                index = None
                gc.collect()
            folder = Path(spec['folder'])
            if measure('verify_vector_bytes', lambda: checksum(folder / 'vecs.f4')) != spec['vector_sha256']:
                raise ValueError('snapshot vector bytes changed')
            with MappedSnapshot(folder, expected_archive_sha256=spec['archive_sha256']) as mapped:
                if (mapped.vectors.shape[1] != dim or mapped.receipt['members']['vecs.f4']['sha256'] != spec['vector_sha256']):
                    raise ValueError('snapshot shape or receipt mismatch')
                current = set(mapped.ids)
                if previous - current or current - labels.keys():
                    raise ValueError('invalid insert-only membership')
                validate_run(reference, list(queries['calibration'].ids) + list(queries[role].ids), mapped.ids, config['k'])
                additions = sorted(current - previous)
                chronological.extend(additions)
                if template is None or mode == 'refreshed_rebuild':
                    template = None
                    template, training = measure('train', lambda: train_template(
                        mapped, config, seed, output / f'training-{step}'))
                    training = {**training, 'snapshot': step}
                if index is None:
                    index = measure('allocate', lambda: faiss.clone_index(template))
                    order = chronological
                else:
                    order = additions
                measure('insert_delta' if mode == 'frozen_incremental' and step > 1 else 'build',
                        lambda: add_mapped(index, mapped, order, labels, config['batch_rows']))
            if index.ntotal != len(current):
                raise ValueError('index live count mismatch')
            membership_hash = measure('verify_membership', lambda: verify_membership(index, ids, current))
            index.nprobe = 1
            state_hash = measure('fingerprint', lambda: streamed_faiss_hash(index))
            sizes = [int(index.invlists.list_size(i)) for i in range(config['nlist'])]
            stage = {'snapshot': step, 'added_count': len(additions), 'live_count': index.ntotal,
                     'live_ids_sha256': membership_hash, 'insertion_order_sha256': digest(chronological),
                     'serialized_index_sha256': state_hash, 'fingerprint_nprobe': 1,
                     'training': training, 'list_sizes': sizes, 'calibration_grid': [], 'runs': {}}
            dest = output / f'snapshot-{step}'
            dest.mkdir()
            cal_ref = {q: reference[q] for q in queries['calibration'].ids}
            score_checks = []
            for probe in config['nprobe_grid']:
                run, detail = search_cohort(index, ids, queries['calibration'], config['k'], probe)
                detail['mean_recall'] = float(np.mean(list(ann_recall(run, cal_ref).values()))) if detail['feasible'] else None
                name = f'calibration.nprobe-{probe}.run'
                export_run(dest / name, run)
                detail.update(file=f'snapshot-{step}/{name}', sha256=checksum(dest / name))
                stage['calibration_grid'].append(detail)
                score_checks.append((run, probe == config['nlist']))
            selection = select_probe(stage['calibration_grid'], config['recall_target'])
            if frozen_probe is None:
                frozen_probe = selection['nprobe']
            stage.update(selection=selection, frozen_snapshot1_nprobe=frozen_probe)
            dev_ref = {q: reference[q] for q in queries[role].ids}
            for regime, probe in (('frozen_snapshot1', frozen_probe), ('current_calibration', selection['nprobe']),
                                  ('all_lists', config['nlist'])):
                run, detail = search_cohort(index, ids, queries[role], config['k'], probe,
                                            repeats=config['search_repeats'])
                if detail['feasible']:
                    validate_run(run, queries[role].ids, current, config['k'])
                values = ann_recall(run, dev_ref)
                detail.update(ann_recall_per_query=values,
                              ann_recall_mean=float(np.mean(list(values.values()))) if detail['feasible'] else None)
                name = f'{role}.{regime}.run'
                export_run(dest / name, run)
                detail.update(file=f'snapshot-{step}/{name}', sha256=checksum(dest / name))
                stage['runs'][regime] = detail
                score_checks.append((run, probe == config['nlist']))
            with MappedSnapshot(folder, expected_archive_sha256=spec['archive_sha256']) as mapped:
                stage['score_verification'] = measure('verify_returned_scores', lambda: verify_returned_scores(
                    mapped, queries, score_checks, reference, batch_rows=config['batch_rows']))
            # Search parameters are serialized; restore them before asserting no index mutation.
            index.nprobe = 1
            if streamed_faiss_hash(index) != state_hash:
                raise ValueError('search changed index state')
            report['snapshots'].append(stage)
            atomic_json(output / 'report.json', report)
            previous = current
            print(json.dumps({**identity, 'stage': 'completed', 'selection': selection}), flush=True)
        report['status'] = 'completed'
        atomic_json(output / 'report.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__)
        atomic_json(output / 'report.json', report)
        raise
    finally:
        del index, template
        gc.collect()


def load_development(inputs, encoded, references):
    inputs, encoded, references = map(Path, (inputs, encoded, references))
    ids, values, raw = load_queries(inputs, encoded)
    if raw['dtype'] != 'float32' or raw['device'] != 'cuda' or raw.get('tf32') is not False:
        raise ValueError('requires declared float32 CUDA development representation')
    contract, _ = validate_inputs(inputs)
    positions = {key: row for row, key in enumerate(ids)}
    queries = {role: SimpleNamespace(ids=keys, vectors=np.ascontiguousarray(values[[positions[q] for q in keys]]))
               for role, keys in contract['roles'].items()}
    refs, receipts = [], []
    for step in (1, 2, 3):
        folder = references / f'snapshot-{step}'
        receipt = json.loads((folder / 'report.json').read_text())
        if (receipt['status'] != 'completed' or receipt['snapshot'] != step or receipt['depth'] != 100
                or receipt['query_inputs_sha256'] != checksum(inputs / 'query-inputs.json')
                or receipt['fp32_receipt_sha256'] != checksum(encoded / 'query-encoding.json')
                or receipt['artifacts']['float32.run'] != checksum(folder / 'float32.run')):
            raise ValueError('exact reference identity mismatch')
        refs.append(read_run(folder / 'float32.run'))
        receipts.append(receipt)
    evidence = {'query_inputs_sha256': checksum(inputs / 'query-inputs.json'),
                'query_encoding_sha256': checksum(encoded / 'query-encoding.json'),
                'exact_reports': {str(s): checksum(references / f'snapshot-{s}/report.json') for s in (1, 2, 3)}}
    return queries, refs, receipts, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('folder', 'audit', 'text_audit', 'inputs', 'encoded', 'references', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('ivf-development.json'))
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_settings(config, args.mode, args.seed)
    if config['k'] != 100 or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('requires depth100 and single-thread environment')
    queries, refs, receipts, evidence = load_development(args.inputs, args.encoded, args.references)
    text = json.loads(args.text_audit.read_text())
    if text['status'] != 'completed' or text['fixed_text_gate_passed'] is not True or [x['snapshot'] for x in text['snapshots']] != [1, 2, 3]:
        raise ValueError('full fixed-text audit required')
    evidence.update(config_sha256=checksum(args.config), text_audit_sha256=checksum(args.text_audit))
    if args.check_only:
        print(json.dumps({'status': 'preflight_passed', 'mode': args.mode, 'seed': args.seed, 'inputs': evidence}))
        return
    audit = json.loads(args.audit.read_text())
    if audit['status'] != 'completed' or audit['fixed_vector_gate_passed'] is not True or [x['snapshot'] for x in audit['snapshots']] != [1, 2, 3]:
        raise ValueError('full fixed-vector audit required')
    specs = []
    for vector, texts, artifact, receipt in zip(audit['snapshots'], text['snapshots'], ARTIFACTS, receipts):
        if (vector['archive_sha256'] != artifact['archive_sha256'] or not vector['scan']['numeric_vectors_valid']
                or texts['conflicting_duplicate_rows'] or texts['dense_coverage']['missing_text_ids']
                or texts['dense_coverage']['unindexed_text_ids']
                or texts['dense_coverage']['id_map_sha256'] != vector['members']['docnos.npids']['sha256']
                or receipt['vector_sha256'] != vector['scan']['vector_sha256']):
            raise ValueError('data audit/reference identity mismatch')
        specs.append({'snapshot': artifact['snapshot'], 'folder': args.folder / f"snapshot-{artifact['snapshot']}",
                      'archive_sha256': artifact['archive_sha256'], 'vector_sha256': vector['scan']['vector_sha256']})
    evidence['vector_audit_sha256'] = checksum(args.audit)
    record = provenance('checkout-build')
    record['development_inputs'] = evidence
    report = ivf_core(specs, queries, refs, args.output, config, mode=args.mode, seed=args.seed, provenance_record=record)
    print(json.dumps({'status': report['status'], 'mode': args.mode, 'seed': args.seed}))


if __name__ == '__main__':
    main()
