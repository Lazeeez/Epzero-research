"""Development screen D5: recall against work along the whole search-width range (design:
docs/screen-d5-extra-backlinks-pareto.md). Needs an Epzero build with `extra_backlinks`.

    python -m research.longeval.pareto_screen SAMPLE OUTPUT.json --order appended --extra-backlinks 8

Aggregates only: recall@10, recall@100, distance computations and single-thread queries per second
at each search width.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .insertion_pilot import insertion_order

WIDTHS = (10, 20, 40, 80, 120, 200, 300, 400, 600, 800)


def recall(found, exact, k):
    return float(np.mean([len(set(found[q, :k].tolist()) & set(exact[q, :k].tolist())) / k for q in range(len(exact))]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('sample', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--order', choices=('appended', 'shuffled', 'reversed'), required=True)
    parser.add_argument('--extra-backlinks', type=int, required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--widths', type=int, nargs='+', default=list(WIDTHS))
    a = parser.parse_args()
    import epzero
    vectors = np.load(a.sample / 'vectors.npy', mmap_mode='r')
    cohort = np.load(a.sample / 'cohort.npy')
    queries = np.ascontiguousarray(np.load(a.sample / 'queries.npy'))
    exact = np.load(a.sample / 'exact-top100.npy')
    metric = json.loads((a.sample / 'sample.json').read_text()).get('metric', 'ip')
    n, dim = vectors.shape
    sequence = insertion_order(np.load(a.sample / 'lexical-rank.npy'), cohort, a.order, a.seed)
    index = epzero.Index(dim=dim, metric=metric, max_elements=n, M=16, ef_construction=100, seed=a.seed,
                         quantization='none', keep_vectors=True, nav_dim=0, fill_links=False, fast_build=False,
                         half_vectors=False, extra_backlinks=a.extra_backlinks)
    started = time.perf_counter()
    for start in range(0, n, 1024):
        chunk = sequence[start:start + 1024]
        index.add(np.ascontiguousarray(vectors[chunk]), labels=chunk.astype(np.int64), threads=1)
    build = time.perf_counter() - started
    rows = []
    for width in a.widths:
        for k in (10, 100):
            if k > width and k == 100:
                continue
            index.search(queries[:20], k=k, ef=max(width, k), rerank=0, rerank_z=0, patience=0, threads=1)  # warm
            started = time.perf_counter()
            labels, _ = index.search(queries, k=k, ef=max(width, k), rerank=0, rerank_z=0, patience=0, threads=1)
            seconds = time.perf_counter() - started
            found = np.asarray(labels, dtype=np.int64)
            work = np.mean([index.trace(queries[q:q + 1], k=k, ef=max(width, k))['distance_computations']
                            for q in range(0, len(queries), max(1, len(queries) // 200))])
            rows.append({'ef': max(width, k), 'k': k, 'recall': recall(found, exact, k),
                         'distance_computations': float(work), 'qps': len(queries) / seconds})
    report = {'kind': 'EXPLORATORY development screen D5 (aggregates only)', 'sample': str(a.sample.name),
              'metric': metric, 'order': a.order, 'seed': a.seed, 'extra_backlinks': a.extra_backlinks,
              'build_seconds': build, 'rows': rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'sample': report['sample'], 'order': a.order, 'eb': a.extra_backlinks, 'build': round(build)}))


if __name__ == '__main__':
    main()
