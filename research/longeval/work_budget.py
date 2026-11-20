"""Full-cohort distance-budget diagnostics on reproduced uncompressed Epzero graphs."""
import argparse
import heapq
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np

from .contracts import digest
from .evaluation import read_run, write_run
from .failure_diagnostic import TARGETS, trace_evidence, trace_run, trace_selection
from .query_diagnostic import validate_inputs
from .release import checksum
from .retention_search import NativeScorer, native_parity, replay_search
from .storage import atomic_json
from .trajectory import run_trajectory

BUDGETS = (1000, 2000, 4000, 8000, 16000, 32000, 64000)
WIDE_EF = 3200


def prefix_searches(trace, start_distance, budgets, k):
    """Top-k at distance-count checkpoints on one native wide-search trajectory.

    This is offline prefix analysis, not a production native stopping API. Native
    uncompressed search scores its upper entry once and layer-0 entry once in
    addition to its emitted neighbor scores. Stable discovery order breaks ties.
    """
    if (k < 1 or not budgets or list(budgets) != sorted(set(budgets))
            or any(type(b) is not int or b < 1 for b in budgets)):
        raise ValueError('invalid distance budgets')
    events = trace['events']
    if trace['distance_computations'] != len(events) + 2:
        raise ValueError('native distance accounting differs from uncompressed trace contract')
    start, upper_count = trace['entry_label'], 0
    layer0 = False
    for level, source, target, distance, accepted in events:
        if level == 0:
            layer0 = True
        else:
            if layer0:
                raise ValueError('upper event follows layer-0 search')
            upper_count += 1
            if accepted:
                start = target
    initial_calls = upper_count + 2
    heap = [(-float(start_distance), 0, int(start))]
    seen = {start}
    sequence = 0
    checkpoints = {}

    def ranking():
        return [(label, -distance) for distance, sequence, label in
                sorted(heap, key=lambda item: (-item[0], -item[1]))]

    if initial_calls in budgets:
        checkpoints[initial_calls] = ranking()
    for ordinal, (level, source, target, distance, accepted) in enumerate(events[upper_count:], 1):
        if level != 0 or target in seen:
            raise ValueError('invalid repeated layer-0 discovery')
        seen.add(target)
        sequence += 1
        item = (-float(distance), -sequence, int(target))
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif distance < -heap[0][0]:
            heapq.heapreplace(heap, item)
        calls = initial_calls + ordinal
        if calls in budgets:
            checkpoints[calls] = ranking()
    final = ranking()
    if final != trace['results']:
        raise ValueError('complete prefix ranking does not match native search')
    total = trace['distance_computations']
    output = {}
    for budget in budgets:
        rows = final if budget >= total else checkpoints.get(budget, [])
        output[budget] = {'results': rows, 'feasible': len(rows) == k,
                          'distance_computations': min(budget, total),
                          'budget': budget, 'full_native_score_stream_covered': budget >= total,
                          'infeasible_reason': None if len(rows) == k else 'budget ends before k layer-0 candidates'}
    return output


def select_budget(points, target):
    eligible = [p for p in points if p['feasible'] and p['mean_recall'] >= target]
    return min((p['budget'] for p in eligible), default=None)


def validate_baseline(args):
    baseline = json.loads(args.baseline.read_text())
    if ((baseline.get('status'), baseline.get('backend'), baseline.get('mode'), baseline.get('seed'))
            != ('completed', 'epzero', 'incremental', args.seed)):
        raise ValueError('incorrect completed baseline')
    config_path = Path(__file__).with_name('trajectory-development.json')
    if json.loads(config_path.read_text()) != baseline['config']:
        raise ValueError('baseline config mismatch')
    contract, _ = validate_inputs(args.inputs)
    trace_selection(contract['roles']['development'], TARGETS)
    for key, path in (('text_audit_sha256', args.text_audit),
                      ('query_inputs_sha256', args.inputs/'query-inputs.json'),
                      ('query_encoding_sha256', args.encoded/'query-encoding.json'),
                      ('config_sha256', config_path)):
        if baseline['provenance']['development_inputs'][key] != checksum(path):
            raise ValueError('baseline input identity mismatch')
    for step in (1, 2, 3):
        folder = args.references/f'snapshot-{step}'
        if baseline['provenance']['development_inputs']['exact_references'][f'snapshot-{step}'] != {
                'report_sha256': checksum(folder/'report.json'), 'run_sha256': checksum(folder/'float32.run')}:
            raise ValueError('baseline exact-reference identity mismatch')
    root = Path(__file__).resolve().parents[2]
    for name, expected in baseline['provenance']['source_files'].items():
        if name != 'research/longeval/trajectory.py' and checksum(root/name) != expected:
            raise ValueError(f'baseline source changed: {name}')
    import epzero._core
    if checksum(epzero._core.__file__) != baseline['provenance']['epzero_binary_sha256']:
        raise ValueError('baseline native binary mismatch')
    for name, version in baseline['provenance']['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'baseline package changed: {name}')
    return baseline, contract


class WorkBudgetObserver:
    def __init__(self, baseline, output, work, *, budgets=BUDGETS, wide_ef=WIDE_EF, targets=TARGETS):
        if ((baseline.get('status'), baseline.get('backend'), baseline.get('mode'))
                != ('completed', 'epzero', 'incremental')
                or [s['snapshot'] for s in baseline['snapshots']] != [1, 2, 3]):
            raise ValueError('requires completed three-snapshot incremental Epzero baseline')
        if (not budgets or list(budgets) != sorted(set(budgets))
                or any(type(b) is not int or b <= 0 for b in budgets)
                or type(wide_ef) is not int or wide_ef <= baseline['config']['ef']):
            raise ValueError('invalid work-budget protocol')
        self.baseline, self.output, self.work = baseline, Path(output), Path(work)
        self.budgets, self.wide_ef, self.targets = tuple(budgets), wide_ef, tuple(targets)
        self.frozen_budget = None
        self.next_snapshot = 1
        self.saved_bytes = 0
        self.output.mkdir(parents=True, exist_ok=False)
        self.work.mkdir(parents=True, exist_ok=False)

    def __call__(self, backend, stage, queries, reference, config):
        step = stage['snapshot']
        if step != self.next_snapshot:
            raise ValueError('work-budget observer requires ordered snapshots')
        prior = self.baseline['snapshots'][step - 1]
        if config != self.baseline['config'] or digest(stage['state']) != digest(prior['state']):
            raise ValueError('baseline graph/state reproduction failed')
        for regime in ('frozen', 'matched_recall'):
            if stage['runs'][regime]['sha256'] != prior['runs'][regime]['sha256']:
                raise ValueError('baseline ranking reproduction failed')
        selected = trace_selection(queries['development'].ids, self.targets)
        dest = self.output/f'snapshot-{step}'
        dest.mkdir()
        work = self.work/f'snapshot-{step}'
        work.mkdir()
        report = {'status': 'running', 'purpose': 'full-cohort offline search-work diagnostic',
            'snapshot': step, 'seed': backend.seed, 'baseline_reproduced': True,
            'baseline_state_sha256': prior['state']['state_sha256'], 'budgets': self.budgets,
            'wide_ef': self.wide_ef, 'native_ef': config['ef'], 'trace_validation_queries': selected,
            'queries_by_role': {role: list(table.ids) for role, table in queries.items()},
            'queries': [], 'cohorts': {}, 'runs': [],
            'limits': ['Offline distance-budget prefixes, not native capped-search latency measurements',
                       'Native graph fingerprint and ordinary search must reproduce',
                       'Calibration selection uses calibration recall only; development grid is descriptive',
                       'Epzero incremental only; not a cross-backend matched-cost comparison',
                       'No final-test queries or relevance judgments']}
        atomic_json(dest/'report.json', report)
        before = backend.state()
        try:
            for role in ('calibration', 'development'):
                table = queries[role]
                ordinary = {ef: backend.search(table, config['k'], ef) for ef in (config['ef'], self.wide_ef)}
                budget_runs = {budget: {} for budget in self.budgets}
                metrics = {budget: [] for budget in self.budgets}
                for number, (q, query) in enumerate(zip(table.ids, table.vectors)):
                    scorer = NativeScorer(backend.index, query)
                    query_result = {'query_id': q, 'role': role, 'native': {}, 'budget_results': {}}
                    for ef in (config['ef'], self.wide_ef):
                        trace = backend.index.trace(query, k=config['k'], ef=ef)
                        if trace_run(backend, trace) != ordinary[ef][q]:
                            raise ValueError('native trace and ordinary ranking mismatch')
                        path = work/f'{role}-{number}.ef-{ef}.jsonl.gz'
                        detail = trace_evidence(backend, trace, reference[q], path, event_cap=2_000_000)
                        if detail['events_truncated'] or path.stat().st_size > 8 * 1024**2:
                            raise ValueError('complete native trace exceeds evidence guard')
                        self.saved_bytes += path.stat().st_size
                        if self.saved_bytes > 384 * 1024**2:
                            raise ValueError('job complete-trace budget exceeds 384 MiB')
                        shutil.copyfile(path, dest/path.name)
                        query_result['native'][str(ef)] = {**detail,
                            'ann_recall': len({d for d, _ in ordinary[ef][q]} & {d for d, _ in reference[q]}) / config['k']}
                        if ef != self.wide_ef:
                            continue
                        start_distance = scorer(detail['layer0_start_label'])
                        prefixes = prefix_searches(trace, start_distance, self.budgets, config['k'])
                        query_result['layer0_start_distance'] = start_distance
                        if q in selected:
                            parity = native_parity(backend.index, query, config['k'], self.wide_ef)
                            for budget, prefix in prefixes.items():
                                if not prefix['feasible']:
                                    continue
                                stopped = replay_search(backend.index.neighbors, scorer,
                                    entry=trace['entry_label'], entry_layer=trace['entry_layer'],
                                    k=config['k'], ef=self.wide_ef, distance_budget=budget)
                                if (stopped['results'] != prefix['results'] or
                                        stopped['distance_computations'] != prefix['distance_computations']):
                                    raise ValueError('offline prefix differs from explicitly stopped traversal')
                            query_result['explicit_stopping_validation'] = {'native_parity': parity['native_parity'],
                                'feasible_budgets_checked': sum(p['feasible'] for p in prefixes.values())}
                        for budget, prefix in prefixes.items():
                            run = trace_run(backend, prefix)
                            budget_runs[budget][q] = run
                            recall = len({doc for doc, _ in run} & {doc for doc, _ in reference[q]}) / config['k']
                            summary = {key: value for key, value in prefix.items() if key != 'results'}
                            summary['ann_recall'] = recall
                            query_result['budget_results'][str(budget)] = summary
                            metrics[budget].append(summary)
                    report['queries'].append(query_result)
                    atomic_json(dest/'report.json', report)
                    if (number + 1) % 10 == 0 or number + 1 == len(table.ids):
                        print(json.dumps({'stage': 'work_budget_queries', 'snapshot': step,
                            'seed': backend.seed, 'role': role, 'completed': number + 1,
                            'total': len(table.ids), 'saved_trace_bytes': self.saved_bytes}), flush=True)
                for ef, run in ordinary.items():
                    name = f'{role}.native-ef{ef}.run'
                    write_run(dest/name, run)
                    if read_run(dest/name) != run:
                        raise ValueError('native run serialization mismatch')
                    if role == 'development' and ef == config['ef'] and checksum(dest/name) != prior['runs']['frozen']['sha256']:
                        raise ValueError('original frozen run changed')
                    report['runs'].append({'role': role, 'kind': 'native', 'ef': ef,
                                           'file': name, 'sha256': checksum(dest/name)})
                points = []
                for budget in self.budgets:
                    name = f'{role}.budget-{budget}.run'
                    write_run(dest/name, budget_runs[budget])
                    # Empty query outputs at infeasible budgets are retained in report.json.
                    reread = read_run(dest/name)
                    if reread != {q: pairs for q, pairs in budget_runs[budget].items() if pairs}:
                        raise ValueError('prefix run serialization mismatch')
                    values = metrics[budget]
                    feasible = all(v['feasible'] for v in values)
                    points.append({'budget': budget, 'query_count': len(values), 'feasible': feasible,
                        'infeasible_query_ids': [q for q, v in zip(table.ids, values) if not v['feasible']],
                        'mean_recall': float(np.mean([v['ann_recall'] for v in values])) if feasible else None,
                        'mean_distances': float(np.mean([v['distance_computations'] for v in values])),
                        'minimum_recall': min(v['ann_recall'] for v in values) if feasible else None,
                        'zero_recall_queries': sum(v['ann_recall'] == 0 for v in values),
                        'full_native_score_stream_queries': sum(v['full_native_score_stream_covered'] for v in values)})
                    report['runs'].append({'role': role, 'kind': 'budget', 'budget': budget,
                        'file': name, 'sha256': checksum(dest/name), 'feasible': feasible})
                report['cohorts'][role] = points
                atomic_json(dest/'report.json', report)
            chosen = select_budget(report['cohorts']['calibration'], config['recall_target'])
            if step == 1:
                self.frozen_budget = chosen
            report['selection'] = {'target': config['recall_target'], 'rule': 'lowest feasible calibration budget meeting mean target',
                'current_calibration_budget': chosen, 'current_target_met': chosen is not None,
                'frozen_snapshot1_budget': self.frozen_budget,
                'frozen_available': self.frozen_budget is not None}
            if backend.state() != before:
                raise ValueError('work-budget diagnostic changed index state')
            report.update(status='completed', state_unchanged=True, cumulative_complete_log_bytes=self.saved_bytes)
            atomic_json(dest/'report.json', report)
            self.next_snapshot += 1
            return {'status': 'completed', 'baseline_reproduced': True,
                    'report_sha256': checksum(dest/'report.json'), 'current_calibration_budget': chosen}
        except BaseException as error:
            report.update(status='failed', error_type=type(error).__name__)
            atomic_json(dest/'report.json', report)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('folder', 'audit', 'text_audit', 'inputs', 'encoded', 'references', 'baseline', 'output', 'work'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--seed', type=int, choices=(11, 29, 47), required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    baseline, contract = validate_baseline(args)
    import epzero
    fixture = epzero.Index(dim=16, metric='ip', max_elements=4, M=2, ef_construction=4)
    fixture.add(np.eye(4, 16, dtype=np.float32), threads=1)
    native_parity(fixture, np.eye(1, 16, dtype=np.float32)[0], 2, 3)
    del fixture
    if args.check_only:
        print(json.dumps({'status': 'validated', 'seed': args.seed, 'native_fixture_parity': True}))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/'work-inputs.json', {'source_sha256': checksum(__file__),
        'baseline_report_sha256': checksum(args.baseline), 'seed': args.seed, 'budgets': BUDGETS,
        'wide_ef': WIDE_EF, 'roles': contract['roles'], 'purpose': 'offline distance-budget development diagnostic'})
    observer = WorkBudgetObserver(baseline, args.output/'work-budget', args.work)
    result = run_trajectory(args.folder, args.audit, args.text_audit, args.inputs, args.encoded,
        args.references, args.output/'trajectory', Path(__file__).with_name('trajectory-development.json'),
        kind='epzero', mode='incremental', seed=args.seed, native_origin='checkout-build', snapshot_observer=observer)
    print(json.dumps({'status': result['status'], 'seed': args.seed, 'snapshots': len(result['snapshots'])}))


if __name__ == '__main__':
    main()
