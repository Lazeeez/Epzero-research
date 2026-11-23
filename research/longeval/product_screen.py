"""Development screen D7: arrival bias in Epzero's product path (design: docs/screen-d7-product-path.md).

    python -m research.longeval.product_screen SAMPLE OUTPUT.json --mode grown --extra-backlinks 4

`grown`: snapshot-1 documents bulk-built with 8-bit codes on all cores and refined, then snapshot 2
and 3 documents inserted one at a time in arrival order. `rebuilt`: everything bulk-built and
refined at once. Aggregates only.
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
    parser.add_argument('--mode', choices=('grown', 'rebuilt'), required=True)
    parser.add_argument('--extra-backlinks', type=int, required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--exact-build', action='store_true', help='fast_build off: build the graph on full vectors')
    a = parser.parse_args()
    import epzero
    vectors = np.load(a.sample / 'vectors.npy', mmap_mode='r')
    cohort = np.load(a.sample / 'cohort.npy')
    queries = np.ascontiguousarray(np.load(a.sample / 'queries.npy'))
    exact = np.load(a.sample / 'exact-top100.npy')
    metric = json.loads((a.sample / 'sample.json').read_text()).get('metric', 'ip')
    n, dim = vectors.shape
    sequence = insertion_order(np.load(a.sample / 'lexical-rank.npy'), cohort, 'appended', a.seed)
    index = epzero.Index(dim=dim, metric=metric, max_elements=n, M=16, ef_construction=100, seed=a.seed,
                         quantization='sq8', keep_vectors=True, nav_dim=0, fill_links=False, fast_build=not a.exact_build,
                         half_vectors=False, extra_backlinks=a.extra_backlinks)
    first = sequence[cohort[sequence] == 1] if a.mode == 'grown' else sequence
    later = sequence[cohort[sequence] != 1] if a.mode == 'grown' else sequence[:0]
    started = time.perf_counter()
    bulk = np.ascontiguousarray(vectors[np.sort(first)])
    index.train(bulk)
    index.add(bulk, labels=np.sort(first).astype(np.int64), threads=0)
    index.refine_links(threads=0, ef=50)               # what epzero.build() does after the bulk build (ef = max(efC/2, 2M))
    bulk_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for start in range(0, len(later), 1024):
        chunk = later[start:start + 1024]
        index.add(np.ascontiguousarray(vectors[chunk]), labels=chunk.astype(np.int64), threads=1)
    insert_seconds = time.perf_counter() - started
    band_of = np.zeros(101, dtype=np.int64)
    for b, (lo, hi) in enumerate(BANDS):
        band_of[lo:hi + 1] = b
    widths = {}
    for width in WIDTHS:
        labels, _ = index.search(queries, k=100, ef=width, rerank=width, threads=1)
        found = np.asarray(labels, dtype=np.int64)
        table = np.zeros((len(queries), 3, len(BANDS), 2), dtype=np.int64)
        for q in range(len(queries)):
            got = set(found[q].tolist())
            for rank, doc in enumerate(exact[q], 1):
                table[q, cohort[doc] - 1, band_of[rank], 0] += 1
                table[q, cohort[doc] - 1, band_of[rank], 1] += doc not in got
        top10, _ = index.search(queries, k=10, ef=width, rerank=width, threads=1)
        top10 = np.asarray(top10, dtype=np.int64)
        recall10 = float(np.mean([len(set(top10[q].tolist()) & set(exact[q, :10].tolist())) / 10 for q in range(len(queries))]))
        work = np.mean([index.trace(queries[q:q + 1], k=100, ef=width)['distance_computations'] for q in range(len(queries))])
        widths[str(width)] = {
            'newest_excess_pp': 100 * risk_difference(table, 3),
            'miss_rate_pct': {str(c): 100 * float(table[:, c - 1, :, 1].sum() / table[:, c - 1, :, 0].sum()) for c in (1, 2, 3)},
            'recall_at_100': float(1 - table[..., 1].sum() / table[..., 0].sum()), 'recall_at_10': recall10,
            'distance_computations_per_query': float(work)}
    report = {'kind': 'EXPLORATORY development screen D7 (aggregates only)', 'mode': a.mode, 'fast_build': not a.exact_build,
              'extra_backlinks': a.extra_backlinks, 'seed': a.seed, 'metric': metric,
              'bulk_seconds': bulk_seconds, 'insert_seconds': insert_seconds, 'later_documents': int(len(later)),
              'widths': widths}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'mode': a.mode, 'eb': a.extra_backlinks, 'ef200': widths['200']}))


if __name__ == '__main__':
    main()
