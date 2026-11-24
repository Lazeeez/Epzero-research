"""Cross-implementation insertion-order builds (screen X1; design: docs/screen-x1-hnswlib.md).

    python -m research.longeval.cross_pilot SAMPLE OUTPUT --order appended --seed 11 --backend hnswlib \
        --efs 100 200 400 800

The same builds and files as `insertion_pilot run` (frozen under extension protocol V1, so not
edited), for implementations it does not cover. `faiss_hnsw` is here only so a test can check that this
module reproduces `insertion_pilot run` exactly. hnswlib exposes no link lists, so its in-degree
fields are null. Aggregates and positions only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .insertion_pilot import BANDS, insertion_order

BACKENDS = ('hnswlib', 'faiss_hnsw')


def _build(backend, vectors, sequence, metric, *, M, ef_construction, seed):
    """An index of the sample in `sequence` order. Returns (search(width, rows) -> sample positions,
    in-degree, out-degree), the degrees per sample position or None."""
    n, dim = vectors.shape
    if backend == 'hnswlib':
        import hnswlib
        index = hnswlib.Index(space='ip' if metric == 'ip' else 'l2', dim=dim)
        index.init_index(max_elements=n, M=M, ef_construction=ef_construction, random_seed=seed)
        index.set_num_threads(1)
        for start in range(0, n, 1024):
            chunk = sequence[start:start + 1024]
            for row, label in zip(np.ascontiguousarray(vectors[chunk]), chunk):   # one row per call keeps the order exact
                index.add_items(row.reshape(1, -1), np.array([label]), num_threads=1)

        def search(width, rows):
            index.set_ef(width)
            labels, _ = index.knn_query(np.ascontiguousarray(rows), k=100, num_threads=1)
            return np.asarray(labels, dtype=np.int64)          # labels are sample positions
        return search, None, None
    import faiss
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2 if metric == 'l2' else faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.rng = faiss.RandomGenerator(seed)
    for start in range(0, n, 1024):
        for row in np.ascontiguousarray(vectors[sequence[start:start + 1024]]):
            index.add(row.reshape(1, -1))

    def search(width, rows):
        index.hnsw.efSearch = width
        _, labels = index.search(np.ascontiguousarray(rows), 100)
        return sequence[labels]                                # storage position -> sample position
    hnsw = index.hnsw
    neighbors, offsets = faiss.vector_to_array(hnsw.neighbors), faiss.vector_to_array(hnsw.offsets)
    width = hnsw.nb_neighbors(0)
    base = np.stack([neighbors[offsets[i]:offsets[i] + width] for i in range(n)])
    indegree, outdegree = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    indegree[sequence] = np.bincount(base[base >= 0], minlength=n)
    outdegree[sequence] = (base >= 0).sum(axis=1)
    return search, indegree, outdegree


def run(sample, output, *, order, seed, backend='hnswlib', ef=200, M=16, ef_construction=100, efs=()):
    """As `insertion_pilot.run`: build one index, score it at `ef` and at every width in `efs`."""
    if backend not in BACKENDS:
        raise ValueError('unknown backend')
    sample, output = Path(sample), Path(output)
    vectors = np.load(sample / 'vectors.npy', mmap_mode='r')
    cohort = np.load(sample / 'cohort.npy')
    queries = np.load(sample / 'queries.npy')
    exact = np.load(sample / 'exact-top100.npy')
    confirm = None
    if (sample / 'confirm-queries.npy').exists():
        confirm = (np.load(sample / 'confirm-queries.npy'), np.load(sample / 'confirm-exact-top100.npy'))
    n = len(vectors)
    metric = json.loads((sample / 'sample.json').read_text()).get('metric', 'ip')
    if metric not in ('ip', 'l2'):
        raise ValueError('unknown metric')
    sequence = insertion_order(np.load(sample / 'lexical-rank.npy'), cohort, order, seed)
    position = np.empty(n, dtype=np.int64)
    position[sequence] = np.arange(n)
    started = time.perf_counter()
    search, indegree, outdegree = _build(backend, vectors, sequence, metric, M=M, ef_construction=ef_construction, seed=seed)
    build_seconds = time.perf_counter() - started
    decile = position * 10 // n

    def score(found, targets):
        table = np.zeros((len(targets), 3, len(BANDS), 2), dtype=np.int64)   # [query, cohort, band, (n, missed)]
        place = np.zeros((len(targets), 10, len(BANDS), 2), dtype=np.int64)  # [query, insertion decile, band, (n, missed)]
        for qi in range(len(targets)):
            returned = set(found[qi].tolist())
            for rank, doc in enumerate(targets[qi], 1):
                band = next(i for i, b in enumerate(BANDS) if b[0] <= rank <= b[1])
                missed = doc not in returned
                table[qi, cohort[doc] - 1, band] += (1, missed)
                place[qi, decile[doc], band] += (1, missed)
        hits = float(np.mean([len(set(found[qi].tolist()) & set(targets[qi].tolist())) / 100 for qi in range(len(targets))]))
        return table, place, hits

    def summary(table, hits):
        return {'recall_at_100': hits,
                'miss_rate': {str(c): float(table[:, c - 1, :, 1].sum() / table[:, c - 1, :, 0].sum()) for c in (1, 2, 3)}}

    found = search(ef, queries)
    per_query, by_position, recall = score(found, exact)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'per-query.npy', per_query)
    np.save(output / 'per-query-position.npy', by_position)
    np.save(output / f'labels-ef{ef}.npy', found.astype(np.int32))
    widths = {}
    for width in efs:
        if width == ef:
            table, place, hits = per_query, by_position, recall
        else:
            labels = search(width, queries)
            table, place, hits = score(labels, exact)
            np.save(output / f'labels-ef{width}.npy', labels.astype(np.int32))
        np.save(output / f'per-query-ef{width}.npy', table)
        np.save(output / f'per-query-position-ef{width}.npy', place)
        widths[str(width)] = summary(table, hits)
    confirmed = {}
    if confirm is not None:
        rows, targets = confirm
        for width in sorted({ef, *efs}):
            labels = search(width, rows)
            table, place, hits = score(labels, targets)
            np.save(output / f'confirm-labels-ef{width}.npy', labels.astype(np.int32))
            np.save(output / f'confirm-per-query-ef{width}.npy', table)
            np.save(output / f'confirm-per-query-position-ef{width}.npy', place)
            confirmed[str(width)] = summary(table, hits)

    def degree(values, c, f):
        return None if values is None else float(f(values[cohort == c]))
    report = {'backend': backend, 'order': order, 'seed': seed, 'ef': ef, 'M': M, 'ef_construction': ef_construction, 'count': int(n),
              'metric': metric, 'build_seconds': build_seconds, 'recall_at_100': recall,
              'cohorts': {str(c): {'documents': int((cohort == c).sum()),
                                   'mean_insertion_position': float(position[cohort == c].mean() / n),
                                   'indegree_mean': degree(indegree, c, np.mean),
                                   'indegree_median': degree(indegree, c, np.median),
                                   'indegree_below_5_share': degree(indegree, c, lambda v: (v < 5).mean()),
                                   'indegree_zero_share': degree(indegree, c, lambda v: (v == 0).mean()),
                                   'outdegree_mean': degree(outdegree, c, np.mean),
                                   'miss_rate': float(per_query[:, c - 1, :, 1].sum() / per_query[:, c - 1, :, 0].sum())}
                          for c in (1, 2, 3)},
              'indegree_by_insertion_decile': None if indegree is None else
              [float(indegree[(position * 10 // n) == d].mean()) for d in range(10)]}
    if backend == 'hnswlib':
        import hnswlib
        report['implementation'] = {'hnswlib': getattr(hnswlib, '__version__', None)}
    if widths:
        report['widths'] = widths
    if confirmed:
        report['confirm'] = {'queries': int(len(confirm[0])), 'widths': confirmed}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('sample', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--order', choices=('appended', 'reversed', 'shuffled'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--backend', choices=BACKENDS, default='hnswlib')
    parser.add_argument('--M', type=int, default=16)
    parser.add_argument('--ef-construction', type=int, default=100)
    parser.add_argument('--efs', type=int, nargs='*', default=())
    a = parser.parse_args()
    r = run(a.sample, a.output, order=a.order, seed=a.seed, backend=a.backend, M=a.M, ef_construction=a.ef_construction, efs=a.efs)
    print(json.dumps({k: r[k] for k in ('backend', 'order', 'seed', 'build_seconds', 'recall_at_100')}))


if __name__ == '__main__':
    main()
