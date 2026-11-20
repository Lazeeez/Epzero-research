"""Small real-vector feasibility experiment, never a full-corpus performance claim."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .frontier_search import frontier_search
from .query_diagnostic import load_queries, validate_inputs
from .release import checksum
from .retention_search import NativeScorer, native_parity
from .storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dense', type=Path, help='existing snapshot-1 directory; no downloads')
    parser.add_argument('inputs', type=Path)
    parser.add_argument('queries', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    import epzero
    import epzero._core

    config = {'sample_rows': 8192, 'sample_seed': 20261118, 'construction_seed': 11,
              'k': 100, 'M': 16, 'ef_construction': 100, 'ef': 200,
              'wide_ef': 3200, 'budgets': [2000, 4000], 'reserve_capacity': 256,
              'shortlist': 8, 'max_resumptions': 8, 'threads': 1}
    ids, queries, _ = load_queries(args.inputs, args.queries)
    contract, _ = validate_inputs(args.inputs)
    source = args.dense / 'vecs.f4'
    if source.stat().st_size != 869902 * 2560 * 4:
        raise ValueError('requires snapshot-1 vector layout')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'purpose': __doc__, 'config': config,
              'sources': {p.name: checksum(p) for p in
                          (Path(__file__), Path(__file__).with_name('frontier_search.py'),
                           Path(__file__).with_name('retention_search.py'))},
              'native_binary_sha256': checksum(epzero._core.__file__),
              'query_encoding_sha256': checksum(args.queries / 'query-encoding.json'),
              'full_corpus_rehashed_in_this_pilot': False,
              'held_out_retrieval_evaluated': False,
              'limits': ['One seed and a small random subset; not the full LongEval graph',
                         'All 100 training queries were already used in development',
                         'Python diagnostic costs cannot establish native latency',
                         'Equal distance caps need not produce equal actual work',
                         'Subset exact references do not measure full-corpus recall']}
    atomic_json(args.output / 'report.json', report)
    rng = np.random.default_rng(config['sample_seed'])
    selected = np.sort(rng.choice(869902, config['sample_rows'], replace=False))
    mapped = np.memmap(source, dtype='<f4', mode='r', shape=(869902, 2560))
    data = np.ascontiguousarray(mapped[selected])
    del mapped
    if not np.isfinite(data).all() or not np.allclose(np.linalg.norm(data, axis=1), 1., atol=1e-5):
        raise ValueError('invalid sample vectors')
    # These are internal row labels, explicitly not external document IDs.
    labels = selected.astype(np.int64)
    np.save(args.output / 'sample-rows.npy', selected, allow_pickle=False)
    report['sample_vectors_sha256'] = hashlib.sha256(data.tobytes()).hexdigest()
    print(json.dumps({'stage': 'build', 'rows': len(data)}), flush=True)
    index = epzero.Index(dim=2560, metric='ip', max_elements=len(data), M=16,
                         ef_construction=100, seed=11)
    index.add(data, labels=labels, threads=1)
    index.save(str(args.output / 'state-before.index'))
    before = checksum(args.output / 'state-before.index')
    # Float64 exact references over precisely the sampled population.
    similarities = queries.astype(np.float64) @ data.astype(np.float64).T
    exact = [list(map(int, labels[np.lexsort((labels, -row))[:100]])) for row in similarities]
    atomic_json(args.output / 'exact.json', dict(zip(ids, exact)))
    groups = defaultdict(list)
    position = {int(label): i for i, label in enumerate(labels)}
    with (args.output / 'queries.jsonl').open('x') as stream:
        for qi, (qid, query) in enumerate(zip(ids, queries)):
            # Entire original trace/ranking/work parity precedes all interventions.
            baseline = native_parity(index, query, 100, 200)
            scorer = NativeScorer(index, query)
            role = 'calibration' if qid in contract['roles']['calibration'] else 'development'
            for budget in config['budgets']:
                for policy, ef in [('off', 200), ('closest', 200), ('overlap', 200), ('off', 3200)]:
                    result = frontier_search(index.neighbors, scorer,
                        entry=baseline['entry_label'], entry_layer=baseline['entry_layer'],
                        k=100, ef=ef, distance_budget=budget, policy=policy,
                        reserve_capacity=256, shortlist=8, max_resumptions=8)
                    if policy == 'off' and ef == 200 and baseline['distance_computations'] < budget:
                        for key in ('results', 'distance_computations', 'nodes_expanded'):
                            if result[key] != baseline[key]:
                                raise ValueError('new search off-mode parity failed')
                    returned = [label for label, _ in result['results']]
                    if len(set(returned)) != 100 or result['distance_computations'] > budget:
                        raise ValueError('invalid result population or budget')
                    scores = [1. - similarities[qi, position[label]] for label in returned]
                    if not np.allclose(scores, [d for _, d in result['results']], atol=2e-6, rtol=0):
                        raise ValueError('independent float64 score check failed')
                    recall = len(set(returned) & set(exact[qi])) / 100
                    record = {'query_id': qid, 'role': role, 'budget': budget,
                              'ann_recall_in_sample': recall, **result}
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    groups[(role, budget, policy, ef)].append(record)
            if (qi + 1) % 10 == 0:
                print(json.dumps({'stage': 'queries', 'completed': qi + 1}), flush=True)
    index.save(str(args.output / 'state-after.index'))
    if checksum(args.output / 'state-after.index') != before:
        raise ValueError('search changed serialized index')
    # Sample indexes are bounded local artifacts, retained to reproduce this pilot.
    summaries = []
    for (role, budget, policy, ef), rows in sorted(groups.items()):
        summaries.append({'role': role, 'budget': budget, 'policy': policy, 'ef': ef,
            'count': len(rows), 'mean_recall': float(np.mean([r['ann_recall_in_sample'] for r in rows])),
            'min_recall': min(r['ann_recall_in_sample'] for r in rows),
            'mean_distances': float(np.mean([r['distance_computations'] for r in rows])),
            'mean_adjacency_ids': float(np.mean([r['adjacency_ids_inspected'] for r in rows])),
            'queries_resumed': sum(bool(r['selections']) for r in rows)})
    report.update(status='completed', graph_unchanged=True, native_parity_queries=100,
                  query_conditions=800, summaries=summaries,
                  artifacts={p.name: checksum(p) for p in args.output.iterdir() if p.name != 'report.json'})
    atomic_json(args.output / 'report.json', report)
    print(json.dumps({'status': 'completed', 'query_conditions': 800, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
