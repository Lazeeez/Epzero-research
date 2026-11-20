"""Native HNSW call timing under the same per-query protocol as the IVF controls."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

import faiss
import numpy as np

from .contracts import digest
from .evaluation import ann_recall, rank, read_run, validate_run, write_run
from .failure_diagnostic import trace_run
from .ivf_trajectory import load_development
from .release import checksum
from .storage import atomic_json
from .trajectory import run_trajectory


def native_cohort(backend, queries, k, effort, repeats=3):
    if backend.kind not in ('epzero', 'faiss_hnsw') or effort < k or repeats != 3:
        raise ValueError('native cost protocol requires HNSW, ef>=k and three repeats')
    faiss.omp_set_num_threads(1)
    if backend.kind == 'faiss_hnsw':
        backend.index.hnsw.efSearch = effort
    call = backend.index.search
    kwargs = dict(k=k, ef=effort, rerank=0, rerank_z=0, patience=0, threads=1) if backend.kind == 'epzero' else dict(k=k)
    lookup = backend.ids if backend.kind == 'epzero' else backend.inserted
    run, measurements = {}, []
    for qid, vector in zip(queries.ids, queries.vectors):
        query = np.ascontiguousarray(vector.reshape(1, -1))
        first = None
        times, counter_records = [], []
        for repeat in range(repeats + 1):
            if backend.kind == 'faiss_hnsw':
                faiss.cvar.hnsw_stats.reset()
            started = time.perf_counter_ns()
            raw = call(query, **kwargs)
            elapsed = time.perf_counter_ns() - started
            if backend.kind == 'epzero':
                labels, distances = raw
                scores = np.float32(1) - distances
            else:
                scores, labels = raw
                counters = {k: int(getattr(faiss.cvar.hnsw_stats, k)) for k in ('ndis', 'nhops')}
                counter_records.append(counters)
            if first is None:
                first = (labels.copy(), scores.copy())
            elif not np.array_equal(first[0], labels) or not np.array_equal(first[1], scores):
                raise ValueError('repeated native search changed results')
            if repeat:
                times.append(elapsed)
        if any(int(label) < 0 or int(label) >= len(lookup) for label in labels[0]):
            raise ValueError('native search returned an invalid label')
        run[qid] = rank([(lookup[int(label)], float(score)) for label, score in zip(labels[0], scores[0])])
        if backend.kind == 'epzero':
            trace = backend.index.trace(vector, k=k, ef=effort)
            if trace_run(backend, trace) != run[qid]:
                raise ValueError('untimed trace disagrees with native query')
            work = {'epzero_distance_computations': int(trace['distance_computations']),
                    'epzero_nodes_expanded': int(trace['nodes_expanded'])}
            del trace
        else:
            if any(c != counter_records[0] for c in counter_records):
                raise ValueError('FAISS work counters changed across repeats')
            work = {'faiss_native_ndis': counter_records[0]['ndis'], 'faiss_native_nhops': counter_records[0]['nhops']}
        measurements.append({'query_id': qid, 'native_call_nanoseconds': times, **work})
    validate_run(run, queries.ids, backend.live, k)
    return run, measurements


class CostObserver:
    def __init__(self, baseline, output):
        if (baseline['status'] != 'completed' or baseline['mode'] != 'incremental'
                or baseline['backend'] not in ('epzero', 'faiss_hnsw')):
            raise ValueError('requires a completed incremental HNSW baseline')
        self.baseline, self.output = baseline, Path(output)
        self.output.mkdir(parents=True, exist_ok=False)

    def __call__(self, backend, stage, queries, reference, config):
        step = stage['snapshot']
        expected = self.baseline['snapshots'][step-1]
        if digest(stage['state']) != digest(expected['state']) or config != self.baseline['config']:
            raise ValueError('baseline state/configuration did not reproduce')
        for regime in ('frozen', 'matched_recall'):
            if stage['runs'][regime]['sha256'] != expected['runs'][regime]['sha256']:
                raise ValueError('baseline ranking did not reproduce')
        destination = self.output / f'snapshot-{step}'
        destination.mkdir()
        original_ef = backend.index.hnsw.efSearch if backend.kind == 'faiss_hnsw' else None
        before = backend.state()
        report = {'status': 'running', 'snapshot': step, 'backend': backend.kind, 'seed': backend.seed,
                  'baseline_state_sha256': expected['state']['state_sha256'], 'runs': {},
                  'protocol': {'query_batch': 1, 'warmups_per_query': 1, 'timed_repeats': 3,
                               'cohort': '50 development queries in original contract order'},
                  'limits': ['Warm native call including binding/keyword overhead, not service throughput',
                             'Trace/validation/state hashing occurs outside query timing',
                             'FAISS ndis/nhops are native counters, not certified identical to Epzero counter semantics',
                             'CPU type and protocol matched to IVF; distinct allocations and fixed within-arm query order',
                             'Development-only; no significance, held-out or general speedup claim']}
        atomic_json(destination / 'report.json', report)
        dev = queries['development']
        refs = {q: reference[q] for q in dev.ids}
        try:
            for regime, effort in (('frozen', config['ef']), ('matched_recall', stage['calibration']['effort']), ('wide', 3200)):
                run, measurements = native_cohort(backend, dev, config['k'], effort)
                path = destination / f'{regime}.run'
                write_run(path, run)
                if read_run(path) != run:
                    raise ValueError('native timing run export changed results')
                if regime != 'wide' and checksum(path) != expected['runs'][regime]['sha256']:
                    raise ValueError('single-query and original batched rankings differ')
                recalls = ann_recall(run, refs)
                report['runs'][regime] = {'effort': effort, 'file': path.name, 'sha256': checksum(path),
                    'ann_recall_per_query': recalls, 'ann_recall_mean': float(np.mean(list(recalls.values()))),
                    'queries': measurements}
                atomic_json(destination / 'report.json', report)
        finally:
            if original_ef is not None:
                backend.index.hnsw.efSearch = original_ef
        if digest(backend.state()) != digest(before):
            raise ValueError('timing or tracing changed index state')
        report['status'] = 'completed'
        atomic_json(destination / 'report.json', report)
        return {'status': 'completed', 'report_sha256': checksum(destination / 'report.json'), 'baseline_reproduced': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('folder', 'audit', 'text_audit', 'inputs', 'encoded', 'references', 'baseline', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    config_path = Path(__file__).with_name('trajectory-development.json')
    if baseline['config'] != json.loads(config_path.read_text()):
        raise ValueError('baseline configuration mismatch')
    if baseline['seed'] not in (11, 29, 47) or baseline['backend'] not in ('epzero', 'faiss_hnsw') or baseline['mode'] != 'incremental':
        raise ValueError('invalid baseline')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('single-thread environment required')
    load_development(args.inputs, args.encoded, args.references)
    p = baseline['provenance']
    for key, path in (('text_audit_sha256', args.text_audit), ('query_inputs_sha256', args.inputs / 'query-inputs.json'),
                      ('query_encoding_sha256', args.encoded / 'query-encoding.json'), ('config_sha256', config_path)):
        if p['development_inputs'][key] != checksum(path):
            raise ValueError('baseline input mismatch')
    for step in (1, 2, 3):
        folder = args.references / f'snapshot-{step}'
        if p['development_inputs']['exact_references'][f'snapshot-{step}'] != {
                'report_sha256': checksum(folder / 'report.json'), 'run_sha256': checksum(folder / 'float32.run')}:
            raise ValueError('baseline exact reference mismatch')
    root = Path(__file__).resolve().parents[2]
    for name, expected in p['source_files'].items():
        if name == 'research/longeval/trajectory.py':
            # Earlier verified diagnostics added this observer hook; both versions are pinned.
            if (expected != 'ecff24ff1571a8024577ff4ebc0bad06f87ec57623b98d6f33642e5ffcedc68a'
                    or checksum(root / name) != '1ee2d416c6ce53b2a103c77d5a12df9e14448185d1d887157d5c24c003b6e739'):
                raise ValueError('unrecognized trajectory observer revision')
            continue
        if checksum(root / name) != expected:
            raise ValueError(f'baseline source changed: {name}')
    import epzero._core
    if checksum(epzero._core.__file__) != p['epzero_binary_sha256']:
        raise ValueError('baseline native binary mismatch')
    if any(importlib.metadata.version(name) != version for name, version in p['packages'].items()):
        raise ValueError('baseline dependency version mismatch')
    if args.check_only:
        print(json.dumps({'status': 'preflight_passed', 'seed': baseline['seed'], 'backend': baseline['backend']}))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / 'inputs.json', {'baseline_report_sha256': checksum(args.baseline), 'source_sha256': checksum(__file__)})
    observer = CostObserver(baseline, args.output / 'native-cost')
    result = run_trajectory(args.folder, args.audit, args.text_audit, args.inputs, args.encoded,
        args.references, args.output / 'trajectory', config_path, kind=baseline['backend'], mode='incremental',
        seed=baseline['seed'], native_origin='checkout-build', snapshot_observer=observer)
    print(json.dumps({'status': result['status'], 'seed': baseline['seed'], 'backend': baseline['backend']}))


if __name__ == '__main__':
    main()
