"""Development screen D11: extra links and the refine pass in Epzero's automatic build
(design: docs/screen-d11-auto-build-speed.md). Needs the Epzero build with Plan.extra_backlinks.

    python -m research.longeval.auto_build_screen SAMPLE OUTPUT.json --refine on --extra-backlinks 4

`epzero.build` chooses everything else. One thread for building and searching. Aggregates only:
build time, then recall@10 / recall@100, queries per second and distance computations per width.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

WIDTHS = (10, 15, 20, 30, 40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800)


def recall(found, exact, k):
    return float(np.mean([len(set(found[q, :k].tolist()) & set(exact[q, :k].tolist())) / k for q in range(len(exact))]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('sample', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--refine', choices=('on', 'off'), required=True)
    parser.add_argument('--extra-backlinks', type=int, required=True)
    a = parser.parse_args()
    import epzero
    vectors = np.ascontiguousarray(np.load(a.sample / 'vectors.npy'))
    queries = np.ascontiguousarray(np.load(a.sample / 'queries.npy'))
    exact = np.load(a.sample / 'exact-top100.npy')
    metric = json.loads((a.sample / 'sample.json').read_text()).get('metric', 'ip')
    started = time.perf_counter()
    index = epzero.build(vectors, metric=metric, threads=1, refine=(a.refine == 'on'), extra_backlinks=a.extra_backlinks)
    build_seconds = time.perf_counter() - started
    plan = index.plan
    sample = range(0, len(queries), max(1, len(queries) // 200))
    rows = []
    for width in WIDTHS:
        for k in (10, 100):
            ef = max(width, k)
            if k == 100 and width < k:
                continue
            index.search(queries[:20], k=k, ef=ef, threads=1)                      # warm
            started = time.perf_counter()
            labels, _ = index.search(queries, k=k, ef=ef, threads=1)
            seconds = time.perf_counter() - started
            work = np.mean([index.trace(queries[q:q + 1], k=k, ef=ef)['distance_computations'] for q in sample])
            rows.append({'ef': ef, 'k': k, 'recall': recall(np.asarray(labels, dtype=np.int64), exact, k),
                         'qps': len(queries) / seconds, 'distance_computations': float(work)})
    report = {'kind': 'EXPLORATORY development screen D11 (aggregates only)', 'sample': a.sample.name, 'metric': metric,
              'refine': a.refine, 'extra_backlinks': a.extra_backlinks, 'build_seconds': build_seconds,
              'plan': {f: getattr(plan, f) for f in ('quantization', 'nav_dim', 'fast_build', 'refine', 'fill_after',
                                                     'rerank', 'extra_backlinks', 'ef', 'M', 'ef_construction')},
              'rows': rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('sample', 'refine', 'extra_backlinks', 'build_seconds')} | {'plan': report['plan']}))


if __name__ == '__main__':
    main()
