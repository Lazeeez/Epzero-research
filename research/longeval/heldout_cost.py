"""Held-out native HNSW latency and search-work counts (FINAL_PROTOCOL_V1, clarification 2).

Rebuilds each held-out incremental HNSW index and requires its states and rankings to
reproduce the analysed held-out runs exactly. Then, outside the primary analysis, it times the
381 test queries with the development native-cost protocol: one warmup and three timed
single-query native calls, one thread. Settings are fixed ef=200, calibration-selected ef, and
a wide diagnostic ef=3200, plus any further fixed widths given with `--efs` (regime `ef<N>`).
Nothing here can change a primary input or output.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .contracts import digest
from .evaluation import ann_recall, read_run, write_run
from .heldout import ROLE, heldout_provenance, load_heldout, snapshot_specs
from .native_cost import native_cohort
from .release import checksum
from .storage import atomic_json

WIDE_EF = 3200
REGIMES = ('frozen', 'matched_recall', 'wide')


class HeldoutCostObserver:
    def __init__(self, baseline, output, efs=()):
        if (baseline.get('status') != 'completed' or baseline.get('mode') != 'incremental'
                or baseline.get('evaluation_role') != ROLE or baseline.get('backend') not in ('epzero', 'faiss_hnsw')):
            raise ValueError('requires a completed held-out incremental HNSW baseline')
        self.baseline, self.output = baseline, Path(output)
        self.efs = tuple(sorted(set(int(ef) for ef in efs) - {WIDE_EF}))
        self.output.mkdir(parents=True, exist_ok=False)

    def __call__(self, backend, stage, queries, reference, config):
        step = stage['snapshot']
        expected = self.baseline['snapshots'][step - 1]
        if digest(stage['state']) != digest(expected['state']) or config != self.baseline['config']:
            raise ValueError('held-out baseline state or configuration did not reproduce')
        for regime in ('frozen', 'matched_recall'):
            if stage['runs'][regime]['sha256'] != expected['runs'][regime]['sha256']:
                raise ValueError('held-out baseline ranking did not reproduce')
        destination = self.output / f'snapshot-{step}'
        destination.mkdir()
        original_ef = backend.index.hnsw.efSearch if backend.kind == 'faiss_hnsw' else None
        before = backend.state()
        report = {'status': 'running', 'snapshot': step, 'backend': backend.kind, 'seed': backend.seed,
                  'baseline_state_sha256': expected['state']['state_sha256'], 'runs': {},
                  'protocol': {'query_batch': 1, 'warmups_per_query': 1, 'timed_repeats': 3,
                               'cohort': f'{len(queries[ROLE].ids)} held-out test queries in contract order'},
                  'limits': ['Warm native call including binding/keyword overhead, not service throughput',
                             'Trace/validation/state hashing occurs outside query timing',
                             'FAISS ndis/nhops are native counters, not certified identical to Epzero counter semantics',
                             'Secondary outcome; cannot alter the primary held-out analysis']}
        atomic_json(destination / 'report.json', report)
        cohort = queries[ROLE]
        refs = {q: reference[q] for q in cohort.ids}
        try:
            for regime, effort in (('frozen', config['ef']), ('matched_recall', stage['calibration']['effort']),
                                   ('wide', WIDE_EF), *((f'ef{ef}', ef) for ef in self.efs)):
                run, measurements = native_cohort(backend, cohort, config['k'], effort)
                path = destination / f'{regime}.run'
                write_run(path, run)
                if read_run(path) != run:
                    raise ValueError('timing run export changed results')
                if regime in ('frozen', 'matched_recall') and checksum(path) != expected['runs'][regime]['sha256']:
                    raise ValueError('single-query and analysed held-out rankings differ')
                recalls = ann_recall(run, refs)
                report['runs'][regime] = {'effort': effort, 'file': path.name, 'sha256': checksum(path),
                                          'ann_recall_per_query': recalls,
                                          'ann_recall_mean': float(np.mean(list(recalls.values()))),
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


def check_baseline(baseline, inputs, config_path):
    """The rebuilt index must use the analysed run's inputs, configuration and native binary."""
    import epzero._core
    _, _, _, evidence = load_heldout(inputs)
    held = baseline['provenance']['heldout_inputs']
    if baseline['config'] != json.loads(Path(config_path).read_text()) or held['config_sha256'] != checksum(config_path):
        raise ValueError('baseline configuration mismatch')
    for key in ('test_inputs_sha256', 'test_encoding_sha256', 'training_query_inputs_sha256', 'training_query_encoding_sha256'):
        if held[key] != evidence[key]:
            raise ValueError(f'baseline input mismatch: {key}')
    if held['exact_references'] != evidence['exact_references']:
        raise ValueError('baseline exact reference mismatch')
    if checksum(epzero._core.__file__) != baseline['provenance']['epzero_binary_sha256']:
        raise ValueError('baseline native binary mismatch')
    return {'status': 'preflight_passed', 'backend': baseline['backend'], 'seed': baseline['seed']}


def run(folder, audit_path, inputs, baseline_path, output, config_path, *, package_root, efs=()):
    from .trajectory import trajectory_core
    baseline = json.loads(Path(baseline_path).read_text())
    check_baseline(baseline, inputs, config_path)
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('single-thread environment required')
    config = json.loads(Path(config_path).read_text())
    queries, refs, receipts, evidence = load_heldout(inputs)
    specs, audits = snapshot_specs(folder, audit_path, inputs, receipts)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'inputs.json', {'baseline_report_sha256': checksum(baseline_path), 'source_sha256': checksum(__file__)})
    record = heldout_provenance(package_root, 'checkout-build')
    record['heldout_inputs'] = {**evidence, **audits, 'config_sha256': checksum(config_path)}
    observer = HeldoutCostObserver(baseline, output / 'native-cost', efs=efs)
    return trajectory_core(specs, queries, refs, output / 'trajectory', config, kind=baseline['backend'],
                           mode='incremental', seed=baseline['seed'], provenance_record=record,
                           snapshot_observer=observer, evaluation_role=ROLE)


def summarize(evidence, output):
    """Per-query median of three calls, then mean and p95 over queries; work counts per query."""
    rows = []
    for path in sorted(Path(evidence).rglob('native-cost/snapshot-*/report.json')):
        report = json.loads(path.read_text())
        if report['status'] != 'completed':
            raise ValueError(f'incomplete timing evidence: {path}')
        for regime, item in report['runs'].items():
            if checksum(path.parent / item['file']) != item['sha256']:
                raise ValueError(f'timing run checksum mismatch: {path}')
            medians = np.array([np.median(q['native_call_nanoseconds']) / 1e6 for q in item['queries']])
            key = 'epzero_distance_computations' if report['backend'] == 'epzero' else 'faiss_native_ndis'
            work = np.array([q[key] for q in item['queries']], dtype=np.float64)
            rows.append({'backend': report['backend'], 'seed': report['seed'], 'snapshot': report['snapshot'],
                         'regime': regime, 'ef': item['effort'], 'queries': len(medians),
                         'recall_mean': item['ann_recall_mean'], 'ms_mean': float(medians.mean()),
                         'ms_p95': float(np.percentile(medians, 95)), 'work_counter': key,
                         'work_mean': float(work.mean()), 'work_p95': float(np.percentile(work, 95))})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'summary.json', {'status': 'completed', 'source_sha256': checksum(__file__), 'rows': rows})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('check')
    for name in ('inputs', 'baseline', 'config'):
        p.add_argument(name, type=Path)
    p = sub.add_parser('run')
    for name in ('folder', 'audit', 'inputs', 'baseline', 'output', 'config'):
        p.add_argument(name, type=Path)
    p.add_argument('--package-root', type=Path, required=True)
    p.add_argument('--efs', type=int, nargs='*', default=(), help='further fixed search widths to time')
    p = sub.add_parser('summary')
    p.add_argument('evidence', type=Path)
    p.add_argument('output', type=Path)
    a = parser.parse_args()
    if a.command == 'check':
        r = check_baseline(json.loads(a.baseline.read_text()), a.inputs, a.config)
    elif a.command == 'run':
        r = run(a.folder, a.audit, a.inputs, a.baseline, a.output, a.config, package_root=a.package_root, efs=a.efs)
        r = {k: r[k] for k in ('status', 'backend', 'mode', 'seed')}
    else:
        r = {'rows': len(summarize(a.evidence, a.output))}
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
