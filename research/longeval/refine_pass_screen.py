"""Development screen D16: a second refine pass in a one-off build (design: docs/screen-d16-second-refine.md).

    python -m research.longeval.refine_pass_screen SAMPLE OUTPUT.json --passes 2 --width 50

Every vector added on all cores with 8-bit codes (`fast_build`, M 16, efConstruction 100 unless
given, no principal-direction walk), then refined `passes` times at `width`. Then a sweep of search widths at
k 10 and 100, one thread: recall, queries per second, distance computations. Aggregates only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .auto_build_screen import WIDTHS, recall


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('sample', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--passes', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--ef-construction', type=int, default=100, help="200 is epzero.build's default (D16b)")
    a = parser.parse_args()
    import epzero
    vectors = np.ascontiguousarray(np.load(a.sample / 'vectors.npy'))
    queries = np.ascontiguousarray(np.load(a.sample / 'queries.npy'))
    exact = np.load(a.sample / 'exact-top100.npy')
    metric = json.loads((a.sample / 'sample.json').read_text()).get('metric', 'ip')
    n, dim = vectors.shape
    index = epzero.Index(dim=dim, metric=metric, max_elements=n, M=16, ef_construction=a.ef_construction, seed=a.seed,
                         quantization='sq8', keep_vectors=True, nav_dim=0, fill_links=False, fast_build=True,
                         half_vectors=False)
    started = time.perf_counter()
    index.train(vectors)
    index.add(vectors, labels=np.arange(n, dtype=np.int64), threads=0)
    add_seconds = time.perf_counter() - started
    refine_seconds = []
    for _ in range(a.passes):
        started = time.perf_counter()
        index.refine_links(threads=0, ef=a.width)
        refine_seconds.append(time.perf_counter() - started)
    sample = range(0, len(queries), max(1, len(queries) // 200))
    rows = []
    for width in WIDTHS:
        for k in (10, 100):
            if k == 100 and width < k:
                continue
            index.search(queries[:20], k=k, ef=width, rerank=width, threads=1)             # warm
            started = time.perf_counter()
            labels, _ = index.search(queries, k=k, ef=width, rerank=width, threads=1)
            seconds = time.perf_counter() - started
            work = np.mean([index.trace(queries[q:q + 1], k=k, ef=width)['distance_computations'] for q in sample])
            rows.append({'ef': width, 'k': k, 'recall': recall(np.asarray(labels, dtype=np.int64), exact, k),
                         'qps': len(queries) / seconds, 'distance_computations': float(work)})
    report = {'kind': 'EXPLORATORY development screen D16 (aggregates only)', 'sample': a.sample.name, 'metric': metric,
              'passes': a.passes, 'width': a.width, 'seed': a.seed, 'ef_construction': a.ef_construction,
              'documents': int(n), 'queries': int(len(queries)),
              'add_seconds': add_seconds, 'refine_seconds': refine_seconds, 'rows': rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('sample', 'passes', 'width', 'add_seconds', 'refine_seconds')}))


if __name__ == '__main__':
    main()
