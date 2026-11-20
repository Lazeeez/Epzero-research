"""Stage-2 sample gates for encoder pseudo-query bridging (PSEUDO_QUERY_PLAN.md).

Rules below were fixed before any bridged graph was searched. Development queries are only
evaluated; nothing is selected on them. Python replay counts distance computations; it is
not native latency. Sample results cannot establish full-corpus behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .frontier_search import frontier_search
from .ngfix_inputs import read_matrix
from .release import checksum
from .storage import atomic_json

DEPTH = 32            # answers per pseudo-query considered for bridging
EXTRA_BUDGET = 4      # maximum added out-edges per node
K = 100
ARM_EFS = (100, 150, 200)
BASE_EFS = (100, 125, 150, 175, 200, 250, 300, 400, 600, 800)
RANDOM_SEED = 20261119


def exact_topk(queries, data, k, *, chunk=512):
    """Float64 top-k positions per query, ties broken by lower position."""
    x = data.astype(np.float64)
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), chunk):
        s = queries[start:start + chunk].astype(np.float64) @ x.T
        for row, values in enumerate(s):
            top = np.argpartition(-values, k)[:k + 1] if k < len(values) else np.arange(len(values))
            # Include every candidate tied with the kth score before ordering.
            cutoff = np.sort(values[top])[::-1][k - 1]
            pool = np.flatnonzero(values >= cutoff)
            out[start + row] = pool[np.lexsort((pool, -values[pool]))][:k]
    return out


def undirected(out_lists):
    adj = [set() for _ in out_lists]
    for u, targets in enumerate(out_lists):
        for v in targets:
            if v != u:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def components(members, adj):
    """Connected components of the induced subgraph, in first-member order."""
    members = list(dict.fromkeys(int(m) for m in members))
    inside, seen, result = set(members), set(), []
    for m in members:
        if m in seen:
            continue
        stack, comp = [m], []
        seen.add(m)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v in inside and v not in seen:
                    seen.add(v)
                    stack.append(v)
        result.append(comp)
    return result


def bridge(base_out, answers, *, budget=EXTRA_BUDGET):
    """Join each pseudo-query's answer components to its top-1 answer.

    ``answers`` rows are ordered best first. For every component not containing the top-1
    answer, its best member v gets edges anchor->v and v->anchor while both endpoints have
    added-edge budget. Components are recomputed on the augmented graph as edges are added.
    """
    added = [[] for _ in base_out]
    adj = undirected(base_out)
    for row in answers:
        row = [int(x) for x in row]
        anchor = row[0]
        rank = {v: i for i, v in enumerate(row)}
        for comp in components(row, adj):
            if anchor in comp:
                continue
            v = min(comp, key=rank.__getitem__)
            if len(added[anchor]) >= budget or len(added[v]) >= budget:
                continue
            if v in base_out[anchor] or v in added[anchor] or anchor in base_out[v] or anchor in added[v]:
                continue
            added[anchor].append(v)
            added[v].append(anchor)
            adj[anchor].add(v)
            adj[v].add(anchor)
    return added


def random_bridge(base_out, edges, *, budget=EXTRA_BUDGET, seed=RANDOM_SEED):
    """Same number of undirected added edges, uniformly random endpoints, same budget."""
    rng = np.random.default_rng(seed)
    added = [[] for _ in base_out]
    n, made, attempts = len(base_out), 0, 0
    while made < edges:
        attempts += 1
        if attempts > 1000 * max(edges, 1):
            raise ValueError('cannot place random control edges under the budget')
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if (u == v or len(added[u]) >= budget or len(added[v]) >= budget or v in base_out[u]
                or v in added[u] or u in base_out[v]):
            continue
        added[u].append(v)
        added[v].append(u)
        made += 1
    return added


def equal_work_gain(base_curve, distances, recall):
    """Arm recall minus base recall linearly interpolated at the arm's distance count."""
    pts = sorted(base_curve)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if not xs[0] <= distances <= xs[-1]:
        return None
    return recall - float(np.interp(distances, xs, ys))


def run(sample_inputs, index_path, encoding, output):
    import epzero

    sample_inputs, encoding, output = map(Path, (sample_inputs, encoding, output))
    data = read_matrix(sample_inputs / 'base.fbin', '<f4')
    queries = read_matrix(sample_inputs / 'queries.fbin', '<f4')
    meta = json.loads((sample_inputs / 'queries.json').read_text())
    labels = json.loads((sample_inputs / 'labels.json').read_text())
    pos = {int(l): i for i, l in enumerate(labels)}
    dev = [i for i, q in enumerate(meta['order']) if q in meta['roles']['development']]
    pseudo = np.load(encoding / 'vectors-A.npy', allow_pickle=False)
    pseudo_ids = json.loads((encoding / 'ids-A.json').read_text())
    forms = {f: np.array([i for i, x in enumerate(pseudo_ids) if x.split(':')[1] == f]) for f in ('title', 'keywords4')}
    index = epzero.Index.load(str(index_path))
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'purpose': __doc__, 'depth': DEPTH, 'extra_budget': EXTRA_BUDGET, 'k': K,
              'arm_efs': ARM_EFS, 'base_efs': BASE_EFS,
              'inputs': {'encoding_sha256': checksum(encoding / 'encoding.json'),
                         'vectors_A_sha256': checksum(encoding / 'vectors-A.npy'),
                         'index_sha256': checksum(index_path), 'source_sha256': checksum(__file__)},
              'deviations': ['Set A titles come from 8,192 corpus documents of which only 40 are in the '
                             'sample (package-01 label-map defect); pseudo-queries act as a query-distribution '
                             'sample, not as each sample document\'s own query',
                             'ef 50 is incompatible with k=100; ef 100/150/200 used'],
              'held_out_retrieval_evaluated': False}
    atomic_json(output / 'report.json', report)

    # Gate 1: in-distribution.
    x64, q64 = data.astype(np.float64), queries[dev].astype(np.float64)
    to_docs = (q64 @ x64.T).max(1)
    gate1 = {'dev_to_documents_top1': float(to_docs.mean())}
    for f, rows in forms.items():
        gate1[f'dev_to_{f}_top1'] = float((q64 @ pseudo[rows].astype(np.float64).T).max(1).mean())
        pt = pseudo[rows].astype(np.float64) @ x64.T
        gate1[f'{f}_to_documents_top1'] = float(pt.max(1).mean())
    gate1['real_query_to_documents_top1'] = float(to_docs.mean())
    gate1['passed'] = any(gate1[f'dev_to_{f}_top1'] > gate1['dev_to_documents_top1'] for f in forms)
    report['gate1_in_distribution'] = gate1

    # Exact references.
    truth = exact_topk(queries, data, K)
    answers = {f: exact_topk(pseudo[rows], data, K) for f, rows in forms.items()}
    answers['docself'] = exact_topk(data, data, K)

    # Gate 2: coverage of development answer occurrences by pseudo-query top-100 unions.
    occurrences = [int(v) for i in dev for v in truth[i]]
    gate2 = {}
    for f in forms:
        union = set(answers[f].ravel().tolist())
        gate2[f] = sum(v in union for v in occurrences) / len(occurrences)
        top10 = set(answers[f][:, :10].ravel().tolist())
        gate2[f + '_top10'] = sum(v in top10 for v in occurrences) / len(occurrences)
    gate2['passed'] = any(gate2[f] >= 0.90 for f in forms)
    report['gate2_coverage'] = gate2

    # Graphs.
    base_out = [[pos[int(v)] for v in index.neighbors(int(l), 0)] for l in labels]
    arms = {'base': [[] for _ in base_out]}
    for f in forms:
        arms[f] = bridge(base_out, answers[f][:, :DEPTH])
    arms['both'] = bridge(base_out, np.vstack([answers['title'][:, :DEPTH], answers['keywords4'][:, :DEPTH]]))
    arms['docself'] = bridge(base_out, answers['docself'][:, :DEPTH])
    edges = {a: sum(map(len, added)) // 2 for a, added in arms.items()}
    arms['random'] = random_bridge(base_out, max(edges[f] for f in forms))
    edges['random'] = sum(map(len, arms['random'])) // 2
    report['added_undirected_edges'] = edges
    report['base_directed_edges'] = sum(map(len, base_out))

    # Gate 3: fragmentation of development answer sets.
    gate3 = {}
    for arm, added in arms.items():
        adj = undirected([b + a for b, a in zip(base_out, added)])
        counts = [len(components(truth[i], adj)) for i in dev]
        gate3[arm] = {'mean_components': float(np.mean(counts)), 'max_components': int(max(counts))}
    gate3['passed'] = any(gate3[f]['mean_components'] < gate3['base']['mean_components'] for f in ('title', 'keywords4', 'both'))
    report['gate3_fragmentation'] = gate3
    atomic_json(output / 'report.json', report)

    # Gate 4: replay search at equal distance work.
    sims = queries.astype(np.float64) @ x64.T
    curves, rows_out = {}, []
    parity = []
    for arm, added in arms.items():
        def neighbors(label, layer, added=added):
            base = index.neighbors(int(label), layer)
            if layer:
                return base
            return list(map(int, base)) + [labels[v] for v in added[pos[int(label)]]]
        for ef in (BASE_EFS if arm == 'base' else ARM_EFS):
            recalls, work = [], []
            for i in dev:
                trace = index.trace(queries[i], k=K, ef=ef)
                score = lambda label, i=i: 1.0 - sims[i, pos[int(label)]]
                result = frontier_search(neighbors, score, entry=int(trace['entry_label']),
                                         entry_layer=int(trace['entry_layer']), k=K, ef=ef, policy='off')
                got = {pos[int(l)] for l, _ in result['results']}
                recall = len(got & set(truth[i].tolist())) / K
                if arm == 'base':
                    native = {pos[int(l)] for l, _ in trace['results']}
                    parity.append(native == got and trace['distance_computations'] == result['distance_computations'])
                recalls.append(recall)
                work.append(result['distance_computations'])
                rows_out.append({'arm': arm, 'ef': ef, 'query': meta['order'][i], 'recall': recall,
                                 'distances': result['distance_computations']})
            curves[(arm, ef)] = (float(np.mean(work)), float(np.mean(recalls)), float(np.min(recalls)))
        print(json.dumps({'arm': arm, 'done': True}), flush=True)
    report['base_native_parity'] = {'checked': len(parity), 'matched': int(sum(parity))}
    base_curve = [(curves[('base', ef)][0], curves[('base', ef)][1]) for ef in BASE_EFS]
    gate4 = {}
    for (arm, ef), (dist, rec, worst) in sorted(curves.items()):
        gate4.setdefault(arm, {})[str(ef)] = {'mean_distances': dist, 'mean_recall': rec, 'min_recall': worst,
            'equal_work_gain': None if arm == 'base' else equal_work_gain(base_curve, dist, rec)}

    def gains(arm):
        return [gate4[arm][str(ef)]['equal_work_gain'] for ef in ARM_EFS]

    def beats(arm, other):
        return all(g is not None and o is not None and g > o for g, o in zip(gains(arm), gains(other)))

    winners = [a for a in ('title', 'keywords4', 'both')
               if all(g is not None and g > 0 for g in gains(a)) and beats(a, 'docself') and beats(a, 'random')]
    gate4['winners'] = winners
    gate4['passed'] = bool(winners)
    gate4['rule'] = ('equal-work gain > 0 at ef 100/150/200 and greater than both the docself and '
                     'random-edge controls at every ef')
    report['gate4_equal_work'] = gate4
    with (output / 'per-query.jsonl').open('x') as stream:
        for row in rows_out:
            stream.write(json.dumps(row) + '\n')
    report['status'] = 'completed'
    report['all_gates_passed'] = all(report[g]['passed'] for g in
        ('gate1_in_distribution', 'gate2_coverage', 'gate3_fragmentation', 'gate4_equal_work'))
    atomic_json(output / 'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sample_inputs', 'index', 'encoding', 'output'):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    r = run(a.sample_inputs, a.index, a.encoding, a.output)
    print(json.dumps({k: r[k] for k in ('all_gates_passed', 'added_undirected_edges', 'base_native_parity')}))


if __name__ == '__main__':
    main()
