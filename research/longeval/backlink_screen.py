"""Development screens D2/D3: Epzero with extra incoming links at insertion (designs:
docs/screen-d2-min-backlinks.md, docs/screen-d3-extra-backlinks.md). D2 used a `min_backlinks` option
that measured no effect and was removed; this script now drives `extra_backlinks`.

    python -m research.longeval.backlink_screen SAMPLE OUTPUT_DIR --order appended --extra-backlinks 4

Writes OUTPUT_DIR/eb{N}/{order}-{seed}/ with report.json and per-query-ef{w}.npy tables in the
insertion pilot's format, so `insertion_strata` can read each eb{N} folder as one implementation.
Aggregates only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .insertion_pilot import BANDS, insertion_order
from .insertion_strata import risk_difference

WIDTHS = (100, 200, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('sample', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--order', choices=('appended', 'shuffled', 'reversed'), required=True)
    parser.add_argument('--extra-backlinks', type=int, required=True)
    parser.add_argument('--seed', type=int, default=11)
    a = parser.parse_args()
    import epzero
    vectors = np.load(a.sample / 'vectors.npy', mmap_mode='r')
    cohort = np.load(a.sample / 'cohort.npy')
    queries = np.ascontiguousarray(np.load(a.sample / 'queries.npy'))
    exact = np.load(a.sample / 'exact-top100.npy')
    n, dim = vectors.shape
    sequence = insertion_order(np.load(a.sample / 'lexical-rank.npy'), cohort, a.order, a.seed)
    position = np.empty(n, dtype=np.int64)
    position[sequence] = np.arange(n)
    decile = position * 10 // n
    index = epzero.Index(dim=dim, metric='ip', max_elements=n, M=16, ef_construction=100, seed=a.seed,
                         quantization='none', keep_vectors=True, nav_dim=0, fill_links=False, fast_build=False,
                         half_vectors=False, extra_backlinks=a.extra_backlinks)
    started = time.perf_counter()
    for start in range(0, n, 1024):
        chunk = sequence[start:start + 1024]
        index.add(np.ascontiguousarray(vectors[chunk]), labels=chunk.astype(np.int64), threads=1)
    build_seconds = time.perf_counter() - started
    indegree = np.zeros(n, dtype=np.int64)
    for label in range(n):
        np.add.at(indegree, np.asarray(index.neighbors(label, 0), dtype=np.int64), 1)
    band_of = np.zeros(101, dtype=np.int64)
    for b, (lo, hi) in enumerate(BANDS):
        band_of[lo:hi + 1] = b
    out = a.output / f'eb{a.extra_backlinks}' / f'{a.order}-{a.seed}'
    out.mkdir(parents=True, exist_ok=False)
    widths = {}
    for width in WIDTHS:
        started = time.perf_counter()
        labels, _ = index.search(queries, k=100, ef=width, rerank=0, rerank_z=0, patience=0, threads=1)
        ms = 1000 * (time.perf_counter() - started) / len(queries)
        labels = np.asarray(labels, dtype=np.int64)
        table = np.zeros((len(queries), 3, len(BANDS), 2), dtype=np.int64)
        for q in range(len(queries)):
            got = set(labels[q].tolist())
            for rank, doc in enumerate(exact[q], 1):
                table[q, cohort[doc] - 1, band_of[rank], 0] += 1
                table[q, cohort[doc] - 1, band_of[rank], 1] += doc not in got
        np.save(out / f'per-query-ef{width}.npy', table)
        computations = np.mean([index.trace(queries[q:q + 1], k=100, ef=width)['distance_computations']
                                for q in range(len(queries))])
        widths[str(width)] = {
            'newest_excess_pp': 100 * risk_difference(table, 3),
            'miss_rate_pct': {str(c): 100 * float(table[:, c - 1, :, 1].sum() / table[:, c - 1, :, 0].sum()) for c in (1, 2, 3)},
            'recall_at_100': float(1 - table[..., 1].sum() / table[..., 0].sum()),
            'distance_computations_per_query': float(computations), 'ms_per_query': ms}
    np.save(out / 'per-query.npy', np.load(out / 'per-query-ef200.npy'))
    report = {'backend': 'epzero-dev', 'order': a.order, 'seed': a.seed, 'ef': 200, 'M': 16, 'ef_construction': 100,
              'extra_backlinks': a.extra_backlinks, 'count': int(n), 'build_seconds': build_seconds,
              'indegree_by_insertion_decile': [float(indegree[decile == d].mean()) for d in range(10)],
              'widths': widths}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('order', 'extra_backlinks', 'build_seconds')} | {'ef200': widths['200']}))


if __name__ == '__main__':
    main()
