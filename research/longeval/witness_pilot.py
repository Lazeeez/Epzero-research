"""Fixed calibration-only repair screen on the previously sealed small graph."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .query_diagnostic import load_queries, validate_inputs
from .release import checksum
from .retention_search import EventLog, NativeScorer, native_parity, replay_search
from .storage import atomic_json
from .witness_repair import Overlay, Trace, propose, select


def main():
    import epzero
    import epzero._core
    root = Path.cwd()
    base = root / 'build/longeval-frontier-pilot-01'
    output = root / 'build/longeval-witness-pilot-01'
    original = json.loads((base / 'report.json').read_text())
    for name in ('state-before.index', 'sample-rows.npy', 'exact.json'):
        if checksum(base / name) != original['artifacts'][name]:
            raise ValueError('base artifact identity mismatch')
    if checksum(epzero._core.__file__) != original['native_binary_sha256']:
        raise ValueError('native runtime changed')
    inputs = root / 'build/longeval-training-query-inputs-01'
    encoding = root / 'build/longeval-pace-query-evidence-7107422/7107422/float32'
    ids, queries, _ = load_queries(inputs, encoding)
    contract, _ = validate_inputs(inputs)
    calibration = contract['roles']['calibration']
    exact = json.loads((base / 'exact.json').read_text())
    fitting_references = {qid: exact[qid] for qid in calibration}
    labels = np.load(base / 'sample-rows.npy', allow_pickle=False)
    index = epzero.Index.load(str(base / 'state-before.index'))
    output.mkdir(exist_ok=False)
    (output / 'traces').mkdir()
    report = {'status': 'fitting', 'purpose': __doc__,
        'base_report_sha256': checksum(base / 'report.json'),
        'plan_sha256': checksum(Path(__file__).with_name('WITNESS_REPAIR_PLAN.md')),
        'sources': {p.name: checksum(p) for p in [Path(__file__), Path(__file__).with_name('witness_repair.py'),
                                                  Path(__file__).with_name('retention_search.py')]},
        'config': {'k': 100, 'ef': 100, 'low_cap': 1000, 'high_ef': 3200, 'high_cap': 4000,
                   'test_caps': [1000, 2000], 'target': .95, 'edge_budget': 32, 'per_source': 2,
                   'random_seed': 20261118},
        'fitting_query_ids': calibration, 'held_out_retrieval_evaluated': False,
        'limits': ['Small sampled graph, one construction seed, previously exposed training queries',
                   'Calibration scores are in-sample after repair fitting',
                   'Python timing is not native method latency',
                   'Exact references are reused; their construction is a learning cost',
                   'No final held-out retrieval or novelty claim']}
    atomic_json(output / 'report.json', report)
    routes, proposals, learning = {}, [], []
    start = time.perf_counter()
    for number, qid in enumerate(calibration):
        query = queries[ids.index(qid)]
        native = native_parity(index, query, 100, 100)
        routes[qid] = (native['entry_label'], native['entry_layer'])
        traces, runs = {}, {}
        record = {'query_id': qid, 'native_parity_distances': native['distance_computations']}
        for name, ef, cap in [('low', 100, 1000), ('high', 3200, 4000)]:
            trace = Trace()
            log = EventLog(output / 'traces' / f'{qid}.{name}.jsonl.gz')
            try:
                def emit(event):
                    trace(event)
                    log(event)
                run = replay_search(index.neighbors, NativeScorer(index, query), entry=routes[qid][0],
                    entry_layer=routes[qid][1], k=100, ef=ef, distance_budget=cap, emit=emit)
            finally:
                receipt = log.close()
            traces[name], runs[name] = trace, run
            record[name] = {'distance_computations': run['distance_computations'],
                            'nodes_expanded': run['nodes_expanded'], 'trace': receipt}
        proposal = propose(qid, traces['low'], traces['high'], runs['low'], fitting_references[qid])
        if proposal is not None:
            proposals.append(proposal)
        record['proposal'] = proposal
        learning.append(record)
        if (number + 1) % 10 == 0:
            print(json.dumps({'stage': 'fit', 'queries': number + 1}), flush=True)
    selected = select(proposals, calibration, edge_budget=32, per_source=2)
    edges = {'original': [], 'witness': [(r['source'], r['destination']) for r in selected],
             'direct_answer': [], 'random': []}
    rng = np.random.default_rng(20261118)
    for row in selected:
        source = row['source']
        occupied = set(map(int, index.neighbors(source, 0))) | {source}
        qid = row['evidence'][0]['query_id']
        used = {dst for src, dst in edges['direct_answer'] if src == source}
        destination = next(v for v in fitting_references[qid] if v not in occupied | used)
        edges['direct_answer'].append((source, destination))
        used = {dst for src, dst in edges['random'] if src == source}
        eligible = [int(v) for v in labels if v not in occupied | used]
        edges['random'].append((source, int(rng.choice(eligible))))
    overlay_record = {'fitting_query_ids': calibration, 'proposals': proposals,
                      'selected': selected, 'edges': edges}
    atomic_json(output / 'overlays.json', overlay_record)
    frozen_overlay_hash = checksum(output / 'overlays.json')
    report.update(status='evaluating', overlay_sha256=frozen_overlay_hash,
        fitting_wall_seconds=time.perf_counter() - start,
        fitting_distance_calls=sum(r['low']['distance_computations'] + r['high']['distance_computations'] for r in learning),
        fitting_validation_distance_calls=sum(r['native_parity_distances'] for r in learning),
        reused_exact_reference_vector_pairs=50 * len(labels), learning=learning,
        proposal_count=len(proposals), selected_edges=len(selected),
        selected_evidence_exact_destination_count=sum(r['evidence'][0]['destination_in_exact'] for r in selected))
    atomic_json(output / 'report.json', report)
    print(json.dumps({'stage': 'overlay_frozen', 'edges': len(selected), 'sha256': frozen_overlay_hash}), flush=True)
    # Development vectors and references enter evaluation only after the overlay is frozen.
    data = np.asarray([index.get_vector(int(label)) for label in labels], dtype=np.float32)
    if hashlib.sha256(data.tobytes()).hexdigest() != original['sample_vectors_sha256']:
        raise ValueError('sample vector identity mismatch')
    similarities = queries.astype(np.float64) @ data.astype(np.float64).T
    positions = {int(v): i for i, v in enumerate(labels)}
    for qi, qid in enumerate(ids):
        independently_sorted = list(map(int, labels[np.lexsort((labels, -similarities[qi]))[:100]]))
        if independently_sorted != exact[qid]:
            raise ValueError('exact reference mismatch')
    overlays = {name: Overlay(index.neighbors, values, labels) for name, values in edges.items()}
    groups = defaultdict(list)
    with (output / 'queries.jsonl').open('x') as stream:
        for qi, (qid, query) in enumerate(zip(ids, queries)):
            if qid not in routes:
                native = native_parity(index, query, 100, 100)
                routes[qid] = (native['entry_label'], native['entry_layer'])
            scorer = NativeScorer(index, query)
            arguments = dict(entry=routes[qid][0], entry_layer=routes[qid][1], k=100)
            wider_cache = {}
            role = 'calibration' if qid in calibration else 'development'
            for cap in (1000, 2000):
                for arm, overlay in overlays.items():
                    overlay.edge_reads = 0
                    adjacency_ids = 0
                    def emit(event):
                        nonlocal adjacency_ids
                        if event[0] == 'neighbors':
                            adjacency_ids += len(event[3])
                    run = replay_search(overlay, scorer, **arguments, ef=100,
                                        distance_budget=cap, emit=emit)
                    actual = run['distance_computations']
                    if actual not in wider_cache:
                        wider_cache[actual] = replay_search(index.neighbors, scorer, **arguments,
                                                           ef=3200, distance_budget=actual)
                    wider = wider_cache[actual]
                    if wider['distance_computations'] != actual:
                        raise ValueError('wider control did not match actual work')
                    for result in (run, wider):
                        got = [label for label, _ in result['results']]
                        if len(got) != len(set(got)) or len(got) != 100:
                            raise ValueError('invalid ranking depth')
                        expected_scores = [1 - similarities[qi, positions[v]] for v in got]
                        if not np.allclose(expected_scores, [d for _, d in result['results']], atol=2e-6, rtol=0):
                            raise ValueError('independent score check failed')
                    recall = len({v for v, _ in run['results']} & set(exact[qid])) / 100
                    wide_recall = len({v for v, _ in wider['results']} & set(exact[qid])) / 100
                    record = {'query_id': qid, 'role': role, 'arm': arm, 'cap': cap,
                        'recall': recall, 'wider_recall_at_same_actual_work': wide_recall,
                        'distances': actual, 'expansions': run['nodes_expanded'],
                        'adjacency_ids': adjacency_ids, 'overlay_edge_reads': overlay.edge_reads,
                        'results': run['results'], 'wider_results': wider['results']}
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    groups[(role, cap, arm)].append(record)
            if (qi + 1) % 20 == 0:
                print(json.dumps({'stage': 'evaluate', 'queries': qi + 1}), flush=True)
    if checksum(output / 'overlays.json') != frozen_overlay_hash:
        raise ValueError('overlay changed during evaluation')
    index.save(str(output / 'base-after.index'))
    if checksum(output / 'base-after.index') != original['artifacts']['state-before.index']:
        raise ValueError('base graph changed')
    summaries = []
    for (role, cap, arm), rows in sorted(groups.items()):
        summaries.append({'role': role, 'cap': cap, 'arm': arm, 'count': len(rows),
            'mean_recall': float(np.mean([r['recall'] for r in rows])),
            'min_recall': min(r['recall'] for r in rows),
            'below_90': sum(r['recall'] < .9 for r in rows),
            'mean_distances': float(np.mean([r['distances'] for r in rows])),
            'mean_wider_recall_at_same_actual_work': float(np.mean([r['wider_recall_at_same_actual_work'] for r in rows])),
            'method_wins_vs_wider': sum(r['recall'] > r['wider_recall_at_same_actual_work'] for r in rows),
            'method_losses_vs_wider': sum(r['recall'] < r['wider_recall_at_same_actual_work'] for r in rows),
            'queries_using_overlay': sum(r['overlay_edge_reads'] > 0 for r in rows)})
    report.update(status='completed', summaries=summaries, query_conditions=800,
        base_graph_unchanged=True, overlay_frozen_before_development=True,
        artifacts={p.relative_to(output).as_posix(): checksum(p) for p in output.rglob('*')
                   if p.is_file() and p.name != 'report.json'})
    atomic_json(output / 'report.json', report)
    print(json.dumps({'status': 'completed', 'selected_edges': len(selected)}), flush=True)


if __name__ == '__main__':
    main()
