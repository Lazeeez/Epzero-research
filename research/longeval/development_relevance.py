"""Development-only evaluator parity on the pinned snapshot-1 training judgments."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytrec_eval

from .evaluation import evaluate, read_run
from .query_diagnostic import validate_inputs
from .release import checksum, manifest_entries, parse_qrels
from .storage import atomic_json


def evaluate_development(release, inputs, exact_folder, output):
    release, inputs, exact_folder, output = map(Path, (release, inputs, exact_folder, output))
    contract, _ = validate_inputs(inputs)
    queries = contract['roles']['development']
    ref = json.loads((exact_folder / 'report.json').read_text())
    if (ref.get('status') != 'completed' or ref.get('snapshot') != 1
            or ref['query_inputs_sha256'] != checksum(inputs / 'query-inputs.json')):
        raise ValueError('requires the matching snapshot-1 query reference')
    entries = manifest_entries(release)
    runs = {}
    for name in ('bfloat16', 'float32'):
        path = exact_folder / (name + '.run')
        if checksum(path) != ref['artifacts'][path.name]:
            raise ValueError('exact run checksum mismatch')
        run = read_run(path)
        expected = set(contract['roles']['development'] + contract['roles']['calibration'])
        if set(run) != expected:
            raise ValueError('exact query population mismatch')
        runs[name] = {q: run[q] for q in queries}
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'purpose': 'development-only relevance and independent evaluator parity',
              'snapshot': 1, 'query_ids': queries, 'query_count': len(queries), 'population': 'explicit development IDs',
              'source_sha256': checksum(__file__), 'exact_report_sha256': checksum(exact_folder / 'report.json'),
              'query_inputs_sha256': checksum(inputs / 'query-inputs.json'),
              'pytrec_eval_version': importlib.metadata.version('pytrec-eval-terrier'), 'sources': {},
              'limits': ['No test judgments or calibration-query relevance scored',
                         'No significance or equivalence claim', 'No historical query-runtime reproduction',
                         'Explicit development population; not official-task population certification']}
    atomic_json(output / 'report.json', report)
    try:
        for family in ('dctr', 'raw'):
            item = entries[f'longeval_sci_training_2026_qrels_{family}']
            path = release / item['cache_path']
            if checksum(path, 'md5') != item['expected_md5']:
                raise ValueError('training qrels identity mismatch')
            all_qrels = parse_qrels(path.read_text(encoding='utf-8-sig'))
            qrels = {q: all_qrels.get(q, {}) for q in queries}
            values = {}
            for precision, run in runs.items():
                ours = evaluate(run, qrels, queries, population='explicit', ndcg_k=10, recall_k=100)
                other = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'recall.100'}).evaluate(
                    {q: dict(pairs) for q, pairs in run.items()})
                for q in queries:
                    for local, external in (('nDCG@10', 'ndcg_cut_10'), ('R@100', 'recall_100')):
                        if not np.isclose(ours['per_query'][q][local], other.get(q, {}).get(external, 0), atol=1e-12, rtol=0):
                            raise ValueError('independent evaluator parity failed')
                values[precision] = ours
            deltas = {q: values['float32']['per_query'][q]['nDCG@10'] - values['bfloat16']['per_query'][q]['nDCG@10'] for q in queries}
            report['sources'][family] = {'qrels_sha256': checksum(path), 'evaluator_parity_passed': True,
                'results': values, 'fp32_minus_bf16_ndcg10': {'mean': float(np.mean(list(deltas.values()))),
                    'changed_queries': sum(value != 0 for value in deltas.values()), 'per_query': deltas}}
            atomic_json(output / 'report.json', report)
        report['status'] = 'completed'
        atomic_json(output / 'report.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__)
        atomic_json(output / 'report.json', report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'inputs', 'exact_folder', 'output'): parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = evaluate_development(args.release, args.inputs, args.exact_folder, args.output)
    print(json.dumps({'status': result['status'], 'query_count': result['query_count'], 'sources': {
        key: {'parity': value['evaluator_parity_passed'], 'means': {p: v['mean'] for p, v in value['results'].items()},
              'mean_ndcg10_delta': value['fp32_minus_bf16_ndcg10']['mean']} for key, value in result['sources'].items()}}, indent=2))
