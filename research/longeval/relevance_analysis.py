"""Score verified development runs against snapshot-1 training judgments only."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytrec_eval

from .evaluation import ann_recall, evaluate, read_run
from .query_diagnostic import validate_inputs
from .release import checksum, manifest_entries, parse_qrels
from .storage import atomic_json


def score(run, reference, qrels, queries):
    if set(run) != set(queries) or set(reference) != set(queries):
        raise ValueError('development query population mismatch')
    for pairs in list(run.values()) + list(reference.values()):
        if len(pairs) != 100 or len({d for d, _ in pairs}) != 100:
            raise ValueError('requires unique depth-100 runs')
    local = evaluate(run, qrels, queries, population='explicit', ndcg_k=10, recall_k=100)
    other = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'recall.100'}).evaluate(
        {q: dict(pairs) for q, pairs in run.items()})
    for q in queries:
        for ours, theirs in (('nDCG@10', 'ndcg_cut_10'), ('R@100', 'recall_100')):
            if not np.isclose(local['per_query'][q][ours], other.get(q, {}).get(theirs, 0), atol=1e-12, rtol=0):
                raise ValueError('independent evaluator parity failed')
    recalls = ann_recall(run, reference)
    for q in queries:
        local['per_query'][q]['ANN_recall@100'] = recalls[q]
    local['mean']['ANN_recall@100'] = float(np.mean(list(recalls.values())))
    return local


def analyze(root, output):
    root, output = Path(root), Path(output)
    build = root / 'build'
    inputs = build / 'longeval-training-query-inputs-01'
    contract, _ = validate_inputs(inputs)
    queries = contract['roles']['development']
    bound = {}

    def checked(path, expected=None):
        path = Path(path)
        actual = checksum(path)
        if expected is not None and actual != expected:
            raise ValueError(f'input identity changed: {path.name}')
        bound[path.relative_to(root).as_posix()] = actual
        return path

    checked(inputs / 'query-inputs.json')
    refs, runs = {}, []
    for step in (1, 2, 3):
        folder = build / 'longeval-query-sensitivity-01' / f'snapshot-{step}'
        receipt = json.loads(checked(folder / 'report.json').read_text())
        if (receipt['status'] != 'completed' or receipt['snapshot'] != step or receipt['depth'] != 100
                or receipt['query_inputs_sha256'] != checksum(inputs / 'query-inputs.json')):
            raise ValueError('exact reference identity mismatch')
        path = checked(folder / 'float32.run', receipt['artifacts']['float32.run'])
        all_queries = read_run(path)
        if set(all_queries) != set(queries + contract['roles']['calibration']):
            raise ValueError('exact query population mismatch')
        refs[step] = {q: all_queries[q] for q in queries}
        runs.append((dict(snapshot=step, backend='exact', mode='exact', seed=0, regime='exact'), refs[step]))
    comparison = json.loads(checked(build / 'longeval-hnsw-three-seeds/comparison.json').read_text())
    for evidence in comparison['evidence']:
        job = str(evidence['job_id'])
        parent = build / f'longeval-pace-trajectory-evidence-{job}'
        verification = json.loads(checked(parent / 'local-verification.json', evidence['verification_sha256']).read_text())
        if verification['status'] != 'verified':
            raise ValueError('unverified trajectory')
        folder = parent / job / 'trajectory'
        report = json.loads(checked(folder / 'report.json', evidence['report_sha256']).read_text())
        if report['status'] != 'completed':
            raise ValueError('incomplete trajectory')
        for stage in report['snapshots']:
            for regime, item in stage['runs'].items():
                path = checked(folder / item['file'], item['sha256'])
                identity = {key: report[key] for key in ('backend', 'mode', 'seed')}
                runs.append(({**identity, 'snapshot': stage['snapshot'], 'regime': regime}, read_run(path)))
    for job in (7130605, 7130606, 7130607):
        parent = build / f'longeval-pace-work-budget-evidence-{job}'
        verification = json.loads(checked(parent / 'local-verification.json').read_text())
        if verification['status'] != 'verified':
            raise ValueError('unverified work-budget experiment')
        folder = parent / str(job)
        for step in (1, 2, 3):
            relative = f'experiment/work-budget/snapshot-{step}/report.json'
            report = json.loads(checked(folder / relative, verification['artifacts'][relative]).read_text())
            for name in ('development.budget-4000.run', 'development.native-ef3200.run'):
                relative = f'experiment/work-budget/snapshot-{step}/{name}'
                item = next(x for x in report['runs'] if x['file'] == name)
                path = checked(folder / relative, verification['artifacts'][relative])
                if checksum(path) != item['sha256']:
                    raise ValueError('work-budget run mismatch')
                runs.append((dict(snapshot=step, backend='epzero', mode='incremental', seed=verification['seed'],
                                  regime='offline_budget4000' if 'budget' in name else 'native_ef3200'), read_run(path)))
    release = build / 'longeval-release/files'
    entries = manifest_entries(release)
    checked(release / 'downloads.json')
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'purpose': 'development relevance against fixed snapshot-1 training judgments',
              'query_count': len(queries), 'query_ids': queries, 'judgment_snapshot': 1,
              'evaluator_version': importlib.metadata.version('pytrec-eval-terrier'),
              'limits': ['Later-snapshot scores use historical training labels, not contemporaneous judgments',
                         'Unjudged additions are not evidence of irrelevance',
                         'No test retrieval outcomes, significance, or generalization claim',
                         'Exact vector neighbors need not maximize judged relevance',
                         'Budgeted rankings are offline wide-search prefixes, not measured native latency'],
              'source_sha256': checksum(__file__), 'summaries': []}
    atomic_json(output / 'report.json', result)
    rows = []
    try:
        for family in ('dctr', 'raw'):
            entry = entries[f'longeval_sci_training_2026_qrels_{family}']
            path = release / entry['cache_path']
            if checksum(path, 'md5') != entry['expected_md5']:
                raise ValueError('training judgment identity mismatch')
            all_qrels = parse_qrels(checked(path).read_text(encoding='utf-8-sig'))
            if set(all_qrels) != set(queries + contract['roles']['calibration']):
                raise ValueError('training judgment query population mismatch')
            qrels = {q: all_qrels[q] for q in queries}
            exact = {step: score(ref, ref, qrels, queries) for step, ref in refs.items()}
            for identity, run in runs:
                step = identity['snapshot']
                evaluated = score(run, refs[step], qrels, queries)
                summary = {**identity, 'family': family, 'judgment_snapshot': 1,
                           'contemporaneous_labels': step == 1, **evaluated['mean']}
                for metric in ('nDCG@10', 'R@100'):
                    summary['ANN_minus_exact_' + metric] = evaluated['mean'][metric] - exact[step]['mean'][metric]
                for q, metrics in evaluated['per_query'].items():
                    rows.append({**identity, 'family': family, 'query_id': q, **metrics,
                        **{'ANN_minus_exact_' + metric: metrics[metric] - exact[step]['per_query'][q][metric]
                           for metric in ('nDCG@10', 'R@100')}})
                result['summaries'].append(summary)
        with (output / 'per-query.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result['summaries'][0]))
            writer.writeheader(); writer.writerows(result['summaries'])
        result.update(status='completed', independent_evaluator_parity=True, run_conditions=len(runs),
                      input_files=bound, artifacts={name: checksum(output / name) for name in ('per-query.csv', 'summary.csv')})
        atomic_json(output / 'report.json', result)
        return result
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__)
        atomic_json(output / 'report.json', result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    result = analyze(args.root, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'query_count', 'run_conditions', 'independent_evaluator_parity')}))
