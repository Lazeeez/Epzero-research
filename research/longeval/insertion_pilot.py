"""EXPLORATORY pilot: does HNSW's miss rate follow insertion position? (post-freeze, sample scale)

On a deterministic sample of snapshot-3 documents that keeps each document's arrival snapshot,
build a single-threaded HNSW index (FAISS or Epzero; M=16, efConstruction=100, inner product) in one of three
insertion orders and compare, per arrival cohort:

  appended   cohort 1, then 2, then 3 (lexical within cohort), as the collection actually grew
  reversed   cohort 3, then 2, then 1
  shuffled   one random permutation

Measured at ef 200 with the 381 test queries: neighbour recall against exact search within the
sample, misses by cohort within exact-rank bands, and base-layer in-degree by cohort. If misses
follow insertion position, the reversed build must move the excess onto cohort 1.

A build can also be searched at further widths (`--efs`) and built with other graph settings
(`--M`, `--ef-construction`), to check that the result does not rest on one setting.

This is a mechanism pilot on a sample, not a result about the full collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

BANDS = ((1, 10), (11, 50), (51, 100))


def prepare(dense, inputs, output, *, count=300_000):
    """Sample documents, store their vectors, cohorts and the exact top-100 of every test query."""
    from .dense import ARTIFACTS
    from .heldout import ROLE, load_heldout
    from .heldout_exploratory import cohorts
    from .mapped import MappedSnapshot
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    first = cohorts(dense)
    with MappedSnapshot(Path(dense) / 'snapshot-3', expected_archive_sha256=ARTIFACTS[2]['archive_sha256']) as snap:
        order = sorted(range(len(snap.ids)), key=lambda i: hashlib.sha256(b'insertion-pilot-v1:' + snap.ids[i].encode()).digest())
        rows = np.sort(np.array(order[:count]))
        ids = [snap.ids[i] for i in rows]
        vectors = np.empty((count, snap.vectors.shape[1]), dtype=np.float32)
        for start in range(0, count, 4096):
            vectors[start:start + 4096] = snap.vectors[rows[start:start + 4096]]
    cohort = np.array([first[d] for d in ids], dtype=np.int8)
    queries, _, _, _ = load_heldout(inputs, references=False)
    q = np.ascontiguousarray(queries[ROLE].vectors)
    scores = np.empty((len(q), count), dtype=np.float32)
    q64 = q.astype(np.float64)
    for start in range(0, count, 20_000):
        scores[:, start:start + 20_000] = (vectors[start:start + 20_000].astype(np.float64) @ q64.T).T
    top = np.argsort(-scores, axis=1, kind='stable')[:, :100].astype(np.int32)
    np.save(output / 'vectors.npy', vectors)
    np.save(output / 'cohort.npy', cohort)
    # Lexical rank of each sampled ID: the study inserts each cohort's additions in lexical ID order.
    lexical = np.empty(count, dtype=np.int64)
    lexical[np.argsort(np.array(ids, dtype=object), kind='stable')] = np.arange(count)
    np.save(output / 'lexical-rank.npy', lexical)
    np.save(output / 'queries.npy', q)
    np.save(output / 'exact-top100.npy', top)
    summary = {'count': count, 'cohort_sizes': {str(c): int((cohort == c).sum()) for c in (1, 2, 3)},
               'queries': len(q), 'ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest()}
    (output / 'sample.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def insertion_order(ids_sorted_rank, cohort, order, seed):
    """Insertion sequence (sample positions) for an order; cohorts are filled in lexical ID order."""
    base = np.lexsort((ids_sorted_rank, cohort))               # cohort, then lexical
    if order == 'appended':
        return base
    if order == 'reversed':
        return np.lexsort((ids_sorted_rank, -cohort.astype(np.int64)))
    if order == 'shuffled':
        return np.random.default_rng(seed).permutation(len(cohort))
    raise ValueError('unknown order')


def run(sample, output, *, order, seed, backend='faiss_hnsw', ef=200, M=16, ef_construction=100, efs=()):
    """Build one index and score it at `ef`; `efs` lists further search widths for the same build."""
    import faiss
    faiss.omp_set_num_threads(1)
    if backend not in ('faiss_hnsw', 'epzero'):
        raise ValueError('unknown backend')
    sample, output = Path(sample), Path(output)
    vectors = np.load(sample / 'vectors.npy', mmap_mode='r')
    cohort = np.load(sample / 'cohort.npy')
    queries = np.load(sample / 'queries.npy')
    exact = np.load(sample / 'exact-top100.npy')
    n, dim = vectors.shape
    sequence = insertion_order(np.load(sample / 'lexical-rank.npy'), cohort, order, seed)
    position = np.empty(n, dtype=np.int64)
    position[sequence] = np.arange(n)                       # insertion position of each sample document
    started = time.perf_counter()
    if backend == 'faiss_hnsw':
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.rng = faiss.RandomGenerator(seed)
        for start in range(0, n, 1024):
            block = np.ascontiguousarray(vectors[sequence[start:start + 1024]])
            for row in block:                               # one row per call keeps the order exact
                index.add(row.reshape(1, -1))
        build_seconds = time.perf_counter() - started

        def search(width):
            index.hnsw.efSearch = width
            _, labels = index.search(np.ascontiguousarray(queries), 100)
            return sequence[labels]                         # storage position -> sample position
        hnsw = index.hnsw
        neighbors = faiss.vector_to_array(hnsw.neighbors)
        offsets = faiss.vector_to_array(hnsw.offsets)
        width = hnsw.nb_neighbors(0)
        base = np.stack([neighbors[offsets[i]:offsets[i] + width] for i in range(n)])
        indegree = np.empty(n, dtype=np.int64)
        indegree[sequence] = np.bincount(base[base >= 0], minlength=n)
        outdegree = np.empty(n, dtype=np.int64)
        outdegree[sequence] = (base >= 0).sum(axis=1)
    else:
        # Same explicit, uncompressed configuration as the study's Epzero adapter (backends.py).
        import epzero
        index = epzero.Index(dim=dim, metric='ip', max_elements=n, M=M, ef_construction=ef_construction, seed=seed,
                             quantization='none', keep_vectors=True, nav_dim=0, fill_links=False,
                             fast_build=False, half_vectors=False)
        for start in range(0, n, 1024):
            chunk = sequence[start:start + 1024]
            index.add(np.ascontiguousarray(vectors[chunk]), labels=chunk.astype(np.int64), threads=1)
        build_seconds = time.perf_counter() - started

        def search(width):
            labels, _ = index.search(np.ascontiguousarray(queries), k=100, ef=width, rerank=0, rerank_z=0, patience=0,
                                     threads=1)
            return np.asarray(labels, dtype=np.int64)       # labels are sample positions
        indegree = np.zeros(n, dtype=np.int64)
        outdegree = np.zeros(n, dtype=np.int64)
        for label in range(n):
            links = np.asarray(index.neighbors(label, 0), dtype=np.int64)
            outdegree[label] = len(links)
            np.add.at(indegree, links, 1)
    def score(found):
        # Misses by cohort and exact-rank band, per query (for a query bootstrap).
        table = np.zeros((len(queries), 3, len(BANDS), 2), dtype=np.int64)   # [query, cohort, band, (n, missed)]
        for qi in range(len(queries)):
            returned = set(found[qi].tolist())
            for rank, doc in enumerate(exact[qi], 1):
                band = next(i for i, b in enumerate(BANDS) if b[0] <= rank <= b[1])
                table[qi, cohort[doc] - 1, band, 0] += 1
                table[qi, cohort[doc] - 1, band, 1] += doc not in returned
        hits = float(np.mean([len(set(found[qi].tolist()) & set(exact[qi].tolist())) / 100 for qi in range(len(queries))]))
        return table, hits

    per_query, recall = score(search(ef))
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'per-query.npy', per_query)
    widths = {}
    for width in efs:
        table, hits = (per_query, recall) if width == ef else score(search(width))
        np.save(output / f'per-query-ef{width}.npy', table)
        widths[str(width)] = {'recall_at_100': hits,
                              'miss_rate': {str(c): float(table[:, c - 1, :, 1].sum() / table[:, c - 1, :, 0].sum())
                                            for c in (1, 2, 3)}}
    report = {'backend': backend, 'order': order, 'seed': seed, 'ef': ef, 'M': M, 'ef_construction': ef_construction, 'count': int(n),
              'build_seconds': build_seconds, 'recall_at_100': recall,
              'cohorts': {str(c): {'documents': int((cohort == c).sum()),
                                   'mean_insertion_position': float(position[cohort == c].mean() / n),
                                   'indegree_mean': float(indegree[cohort == c].mean()),
                                   'indegree_median': float(np.median(indegree[cohort == c])),
                                   'indegree_below_5_share': float((indegree[cohort == c] < 5).mean()),
                                   'indegree_zero_share': float((indegree[cohort == c] == 0).mean()),
                                   'outdegree_mean': float(outdegree[cohort == c].mean()),
                                   'miss_rate': float(per_query[:, c - 1, :, 1].sum() / per_query[:, c - 1, :, 0].sum())}
                          for c in (1, 2, 3)},
              # In-degree by insertion decile, whatever the cohort.
              'indegree_by_insertion_decile': [float(indegree[(position * 10 // n) == d].mean()) for d in range(10)]}
    if widths:
        report['widths'] = widths
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def _ratio(total, target, reference):
    """Observed misses of the target cohorts over the misses expected at the reference cohort's
    miss rate in each exact-rank band. `total` is [cohort, band, (n, missed)]; cohorts are 1-based."""
    target = [c - 1 for c in ((target,) if isinstance(target, int) else target)]
    observed = total[target, :, 1].sum()
    expected = sum(total[target, b, 0].sum() * total[reference - 1, b, 1] / max(1, total[reference - 1, b, 0])
                   for b in range(len(BANDS)))
    return float(observed / expected) if expected else float('nan')   # undefined without reference misses


def _draws(queries, draws, seed):
    """The same query resamples for every build, so that builds can be compared within a draw."""
    return np.random.default_rng(seed).integers(0, queries, (draws, queries))


def ratio(per_query, target, reference, *, draws=5000, seed=20261120):
    """Rank-standardized miss ratio of cohort(s) `target` vs cohort `reference`, with a query bootstrap."""
    boot = [_ratio(per_query[rows].sum(axis=0), target, reference) for rows in _draws(len(per_query), draws, seed)]
    return {'ratio': float(_ratio(per_query.sum(axis=0), target, reference)),
            'interval_95': np.quantile(boot, [.025, .975]).tolist()}


def _table(folder, ef=None):
    """Per-query miss table of one build, at its primary width or at one of its further widths."""
    return np.load(Path(folder) / ('per-query.npy' if ef is None else f'per-query-ef{ef}.npy'))


def summarize(root, ef=None):
    out = []
    for path in sorted(Path(root).glob('*/report.json')):
        report = json.loads(path.read_text())
        report.setdefault('backend', 'faiss_hnsw')          # the first pilot's reports predate the field
        per_query = _table(path.parent, ef)
        if ef is not None:
            report['ef'] = ef
            report['recall_at_100'] = report['widths'][str(ef)]['recall_at_100']
        total = per_query.sum(axis=0)
        # Numerators and denominators behind every ratio: [cohort][band] = exact neighbours, missed.
        report['counts'] = {str(c): {f'{lo}-{hi}': {'exact_neighbours': int(total[c - 1, b, 0]), 'missed': int(total[c - 1, b, 1])}
                                     for b, (lo, hi) in enumerate(BANDS)} for c in (1, 2, 3)}
        report['miss_ratio_cohort3_vs_cohort1'] = ratio(per_query, 3, 1)
        report['miss_ratio_cohort2_vs_cohort1'] = ratio(per_query, 2, 1)
        report['miss_ratio_later_vs_cohort1'] = ratio(per_query, (2, 3), 1)
        out.append(report)
    return out


def contrast(root, *, ef=None, draws=5000, seed=20261120):
    """Paired comparison of insertion orders on the same queries.

    For each seed and for the mean over seeds: the miss ratio in each order, and the as-arrived
    ratio minus the reversed ratio. Every build is resampled with the same query draws, so the
    interval of the difference is paired.
    """
    builds = {}
    for path in sorted(Path(root).glob('*/report.json')):
        report = json.loads(path.read_text())
        builds[(report['order'], report['seed'])] = _table(path.parent, ef)
    seeds = sorted({seed_ for _, seed_ in builds})
    rows = _draws(len(next(iter(builds.values()))), draws, seed)
    out = {}
    for name, target in (('cohort3_vs_cohort1', 3), ('cohort2_vs_cohort1', 2), ('later_vs_cohort1', (2, 3))):
        point = {key: _ratio(per_query.sum(axis=0), target, 1) for key, per_query in builds.items()}
        boot = {key: np.array([_ratio(per_query[r].sum(axis=0), target, 1) for r in rows]) for key, per_query in builds.items()}

        def item(value, sample):
            return {'value': float(value), 'interval_95': np.quantile(sample, [.025, .975]).tolist()}
        block = {'by_seed': {}, 'seed_mean': {}}
        for s in seeds:
            block['by_seed'][str(s)] = {
                'as_arrived_minus_reversed': item(point[('appended', s)] - point[('reversed', s)],
                                                  boot[('appended', s)] - boot[('reversed', s)]),
                'as_arrived_minus_shuffled': item(point[('appended', s)] - point[('shuffled', s)],
                                                  boot[('appended', s)] - boot[('shuffled', s)])}
        mean = {order: (np.mean([point[(order, s)] for s in seeds]), np.mean([boot[(order, s)] for s in seeds], axis=0))
                for order in ('appended', 'shuffled', 'reversed')}
        for order, (value, sample) in mean.items():
            block['seed_mean'][order] = item(value, sample)
        block['seed_mean']['as_arrived_minus_reversed'] = item(mean['appended'][0] - mean['reversed'][0],
                                                               mean['appended'][1] - mean['reversed'][1])
        block['seed_mean']['as_arrived_minus_shuffled'] = item(mean['appended'][0] - mean['shuffled'][0],
                                                               mean['appended'][1] - mean['shuffled'][1])
        block['seed_mean']['shuffled_minus_reversed'] = item(mean['shuffled'][0] - mean['reversed'][0],
                                                             mean['shuffled'][1] - mean['reversed'][1])
        out[name] = block
    return out


def robustness(sample, roots, output):
    """Aggregates of builds searched at several widths: {root name: {backend: {ef: ...}}}.

    For each width: recall, the seed-mean miss ratios of each order with paired contrasts, and the
    missed neighbours by cohort. Counts and ratios only.
    """
    out = {'kind': 'EXPLORATORY insertion-order pilot: graph settings and search widths (aggregates only)',
           'sample': json.loads((Path(sample) / 'sample.json').read_text()),
           'bootstrap': {'draws': 5000, 'seed': 20261120, 'unit': 'query'}, 'settings': {}}
    for root in map(Path, roots):
        reports = [json.loads(path.read_text()) for path in sorted(root.glob('*/*/report.json'))]   # root/backend/build
        widths = sorted({int(w) for r in reports for w in r['widths']})
        block = {'M': reports[0]['M'], 'ef_construction': reports[0]['ef_construction'], 'backends': {}}
        if any((r['M'], r['ef_construction']) != (block['M'], block['ef_construction']) for r in reports):
            raise ValueError(f'mixed graph settings in {root}')
        for backend in sorted({r['backend'] for r in reports}):
            mine = [r for r in reports if r['backend'] == backend]
            folder = root / backend
            per_width = {}
            for ef in widths:
                builds = summarize(folder, ef)
                per_width[str(ef)] = {
                    'recall_at_100': {order: [b['recall_at_100'] for b in builds if b['order'] == order]
                                      for order in ('appended', 'shuffled', 'reversed')},
                    'missed_by_cohort': {f"{b['order']}-{b['seed']}": {c: sum(v['missed'] for v in b['counts'][c].values())
                                                                     for c in '123'} for b in builds},
                    'contrasts': contrast(folder, ef=ef)}
            block['backends'][backend] = {
                'builds': len(mine), 'build_minutes_median': float(np.median([r['build_seconds'] for r in mine]) / 60),
                'indegree_by_insertion_decile': {f"{r['order']}-{r['seed']}": r['indegree_by_insertion_decile'] for r in mine},
                'widths': per_width}
        out['settings'][root.name] = block
    Path(output).write_text(json.dumps(out, indent=2) + '\n')
    return out


def aggregate(sample, roots, output):
    """One aggregate file for the pilot: sample identity, per-build counts and ratios, paired contrasts.

    It holds counts and ratios only, no identifiers, vectors or rankings.
    """
    out = {'kind': 'EXPLORATORY insertion-order pilot on a sample (aggregates only)',
           'sample': json.loads((Path(sample) / 'sample.json').read_text()),
           'bands': [f'{lo}-{hi}' for lo, hi in BANDS], 'bootstrap': {'draws': 5000, 'seed': 20261120, 'unit': 'query'},
           'builds': [], 'contrasts': {}}
    for root in roots:
        builds = summarize(root)
        out['builds'].extend(builds)
        out['contrasts'][builds[0]['backend']] = contrast(root)
    Path(output).write_text(json.dumps(out, indent=2) + '\n')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('dense', 'inputs', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--count', type=int, default=300_000)
    p = sub.add_parser('run')
    p.add_argument('sample', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--order', choices=('appended', 'reversed', 'shuffled'), required=True)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--backend', choices=('faiss_hnsw', 'epzero'), default='faiss_hnsw')
    p.add_argument('--M', type=int, default=16)
    p.add_argument('--ef-construction', type=int, default=100)
    p.add_argument('--efs', type=int, nargs='*', default=(), help='further search widths for the same build')
    p = sub.add_parser('summary')
    p.add_argument('root', type=Path)
    p = sub.add_parser('aggregate')
    p.add_argument('sample', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('roots', type=Path, nargs='+')
    p = sub.add_parser('robustness')
    p.add_argument('sample', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('roots', type=Path, nargs='+')
    a = parser.parse_args()
    if a.command == 'prepare':
        r = prepare(a.dense, a.inputs, a.output, count=a.count)
    elif a.command == 'run':
        r = run(a.sample, a.output, order=a.order, seed=a.seed, backend=a.backend, M=a.M,
                ef_construction=a.ef_construction, efs=tuple(a.efs))
        r = {k: r[k] for k in ('order', 'seed', 'build_seconds', 'recall_at_100', 'cohorts', 'widths') if k in r}
    elif a.command == 'aggregate':
        r = aggregate(a.sample, a.roots, a.output)['contrasts']
    elif a.command == 'robustness':
        r = {name: {b: list(v['widths']) for b, v in block['backends'].items()}
             for name, block in robustness(a.sample, a.roots, a.output)['settings'].items()}
    else:
        r = summarize(a.root)
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
