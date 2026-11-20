"""Post-hoc one-candidate deferred-expansion experiment with native parity gates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .contracts import digest
from .failure_diagnostic import TARGETS, trace_run, trace_selection
from .release import checksum
from .retention_search import EventLog, NativeScorer, native_parity, replay_search
from .storage import atomic_json
from .trajectory import run_trajectory

BRIDGES = ('85864826', '154424178')
LOW_EFFORTS = (200, 1600)
BASELINE_EFFORTS = (200, 201, 400, 1600, 1601, 3200)


class RetentionObserver:
    def __init__(self, baseline, output, work, *, targets=TARGETS, bridges=BRIDGES,
                 low_efforts=LOW_EFFORTS, baseline_efforts=BASELINE_EFFORTS):
        if (baseline.get('status') != 'completed' or baseline.get('backend') != 'epzero'
                or baseline.get('mode') != 'incremental'
                or [s['snapshot'] for s in baseline['snapshots']] != [1, 2, 3]):
            raise ValueError('requires completed three-snapshot Epzero incremental baseline')
        self.baseline = baseline
        self.output, self.work = Path(output), Path(work)
        self.targets, self.bridges = tuple(targets), tuple(bridges)
        self.low_efforts, self.baseline_efforts = tuple(low_efforts), tuple(baseline_efforts)
        if (len(self.bridges) != 2 or len(set(self.bridges)) != 2
                or not self.low_efforts or min(self.baseline_efforts) < baseline['config']['k']
                or not set(self.low_efforts) <= set(self.baseline_efforts)
                or not {ef + 1 for ef in self.low_efforts} <= set(self.baseline_efforts)
                or max(self.baseline_efforts) <= max(self.low_efforts)):
            raise ValueError('invalid retention protocol grid or bridges')
        self.output.mkdir(parents=True, exist_ok=False)
        self.work.mkdir(parents=True, exist_ok=False)

    def __call__(self, backend, stage, queries, reference, config):
        step = stage['snapshot']
        expected = self.baseline['snapshots'][step - 1]
        if config != self.baseline['config'] or digest(stage['state']) != digest(expected['state']):
            raise ValueError('baseline graph/state reproduction failed')
        for regime in ('frozen', 'matched_recall'):
            if stage['runs'][regime]['sha256'] != expected['runs'][regime]['sha256']:
                raise ValueError('baseline ranking reproduction failed')
        if step != 3:
            return {'status': 'baseline_reproduced', 'intervention': False}
        if not set(self.bridges) <= backend.live:
            raise ValueError('bridge document not live')
        selected = trace_selection(queries['development'].ids, self.targets)
        dest = self.output / 'snapshot-3'
        dest.mkdir()
        report = {'status': 'running', 'purpose': 'post-hoc selected-candidate sufficiency diagnostic',
                  'snapshot': step, 'seed': backend.seed, 'baseline_reproduced': True,
                  'baseline_state_sha256': expected['state']['state_sha256'],
                  'targets': list(self.targets), 'controls': selected[len(self.targets):],
                  'bridges': {key: backend.labels[key] for key in self.bridges},
                  'low_efforts': self.low_efforts, 'baseline_efforts': self.baseline_efforts,
                  'arms': [], 'full_event_logs': True,
                  'limits': ['Post-hoc candidate identities; not a deployable selection rule',
                             'Diagnostic Python runtime is not a native latency benchmark',
                             'Equal distance-evaluation caps do not equate expansions, memory or time',
                             'Only four selected development queries, not a prevalence estimate']}
        atomic_json(dest / 'report.json', report)
        before = backend.state()
        positions = {q: i for i, q in enumerate(queries['development'].ids)}
        saved_bytes = 0

        def record(name, query_id, execute, **metadata):
            nonlocal saved_bytes
            path = self.work / f'{name}.jsonl.gz'
            log = EventLog(path, max_bytes=8 * 1024**2)
            try:
                result = execute(log)
            finally:
                receipt = log.close()
            if not receipt['complete']:
                raise ValueError('incomplete log cannot support experimental result')
            saved_bytes += path.stat().st_size
            if saved_bytes > 256 * 1024**2:
                raise ValueError('job evidence exceeds 256 MiB compressed-log budget')
            shutil.copyfile(path, dest / path.name)
            run = trace_run(backend, result)
            exact_ids = {doc for doc, _ in reference[query_id]}
            recall = len(exact_ids & {doc for doc, _ in run}) / config['k']
            row = {key: value for key, value in result.items()
                   if key not in ('results', 'evicted_unexpanded')}
            row.update(name=name, query_id=query_id, **metadata, event_log=receipt,
                       run=run, ann_recall=recall,
                       evicted_unexpanded_count=len(result['evicted_unexpanded']),
                       evicted_unexpanded_sha256=digest(result['evicted_unexpanded']))
            report['arms'].append(row)
            atomic_json(dest / 'report.json', report)
            print(json.dumps({'stage': 'retention_arm', 'seed': backend.seed,
                              'name': name, 'recall': recall,
                              'distance_computations': result['distance_computations']}), flush=True)
            return result

        try:
            for number, q in enumerate(selected):
                query = queries['development'].vectors[positions[q]]
                baselines = {}
                for ef in self.baseline_efforts:
                    baselines[ef] = record(f'q{number}.native-ef{ef}', q,
                        lambda log, ef=ef: native_parity(backend.index, query, config['k'], ef, emit=log),
                        arm='native_replay', rescue_document=None)
                # All native baselines pass before any intervention on this query.
                route = baselines[self.low_efforts[0]]
                if any((r['entry_label'], r['entry_layer'], r['layer0_start'], r['upper_events']) !=
                       (route['entry_label'], route['entry_layer'], route['layer0_start'], route['upper_events'])
                       for r in baselines.values()):
                    raise ValueError('baseline upper-layer route changes with effort')
                scorer = NativeScorer(backend.index, query)
                def run(ef, log, rescue=None, budget=None):
                    result = replay_search(backend.index.neighbors, scorer,
                        entry=route['entry_label'], entry_layer=route['entry_layer'],
                        k=config['k'], ef=ef, rescue=rescue, distance_budget=budget, emit=log)
                    if (result['layer0_start'], result['upper_events']) != (route['layer0_start'], route['upper_events']):
                        raise ValueError('intervention changed upper-layer route')
                    result.update(entry_label=route['entry_label'], entry_layer=route['entry_layer'],
                                  native_binary_sha256=scorer.binary_sha256)
                    return result
                for low in self.low_efforts:
                    # Predeclared hash choice: no query relevance or intervention result used.
                    candidates = [label for label in baselines[low]['evicted_unexpanded']
                                  if backend.ids[label] not in self.bridges]
                    sham = min(candidates, key=lambda label: hashlib.sha256(
                        f'longeval-retention-sham-v1:{backend.seed}:{q}:{low}:{backend.ids[label]}'.encode()).digest(),
                        default=None)
                    choices = [(f'bridge{i + 1}', backend.labels[key]) for i, key in enumerate(self.bridges)]
                    choices.append(('sham', sham))
                    for tag, label in choices:
                        name = f'q{number}.ef{low}.{tag}'
                        result = record(name, q, lambda log, label=label: run(low, log, rescue=label),
                            arm='deferred_expansion', rescue_document=None if label is None else backend.ids[label],
                            sham_available=sham is not None if tag == 'sham' else None)
                        if not result['rescue_triggered']:
                            for key in ('results', 'distance_computations', 'nodes_expanded'):
                                if result[key] != baselines[low][key]:
                                    raise ValueError('inactive rescue changed search')
                        record(name + '.work-control', q,
                            lambda log: run(max(self.baseline_efforts), log, budget=result['distance_computations']),
                            arm='wide_beam_distance_cap', paired_intervention=name,
                            rescue_document=None)
            if backend.state() != before:
                raise ValueError('retention experiment changed graph state')
            report.update(status='completed', state_unchanged=True, log_bytes=saved_bytes)
            atomic_json(dest / 'report.json', report)
            return {'status': 'completed', 'report_sha256': checksum(dest / 'report.json'),
                    'baseline_reproduced': True, 'arms': len(report['arms'])}
        except BaseException as error:
            report.update(status='failed', error_type=type(error).__name__)
            atomic_json(dest / 'report.json', report)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ('folder', 'audit', 'text_audit', 'inputs', 'encoded', 'references', 'baseline', 'output', 'work')
    for name in names:
        parser.add_argument(name, type=Path)
    parser.add_argument('--seed', type=int, choices=(11, 29), required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    # Reuse the original fail-closed source, binary, package, query and exact-reference checks.
    subprocess.run([sys.executable, '-m', 'research.longeval.failure_diagnostic',
                    *[str(getattr(args, name)) for name in names],
                    '--seed', str(args.seed), '--check-only'], check=True)
    # This fixture checks the exported-kernel ABI before any corpus download.
    import epzero
    import numpy as np
    fixture = epzero.Index(dim=16, metric='ip', max_elements=4, M=2, ef_construction=4)
    fixture.add(np.eye(4, 16, dtype=np.float32), threads=1)
    native_parity(fixture, np.eye(1, 16, dtype=np.float32)[0], 2, 3)
    del fixture
    if args.check_only:
        print(json.dumps({'status': 'validated', 'native_parity_fixture': True, 'seed': args.seed}))
        return
    baseline = json.loads(args.baseline.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / 'retention-inputs.json', {
        'baseline_report_sha256': checksum(args.baseline), 'seed': args.seed,
        'source_files': {str(path): checksum(path) for path in (
            Path(__file__), Path(__file__).with_name('retention_search.py'))},
        'targets': TARGETS, 'bridges': BRIDGES, 'low_efforts': LOW_EFFORTS,
        'baseline_efforts': BASELINE_EFFORTS})
    observer = RetentionObserver(baseline, args.output / 'retention', args.work)
    result = run_trajectory(args.folder, args.audit, args.text_audit, args.inputs, args.encoded,
        args.references, args.output / 'trajectory', Path(__file__).with_name('trajectory-development.json'),
        kind='epzero', mode='incremental', seed=args.seed, native_origin='checkout-build', snapshot_observer=observer)
    print(json.dumps({'status': result['status'], 'seed': args.seed, 'purpose': 'retention intervention'}))


if __name__ == '__main__':
    main()
