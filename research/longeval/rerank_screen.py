"""Development screen D12: the recall ceiling of the principal-direction walk
(design: docs/screen-d12-rerank-ceiling.md).

    python -m research.longeval.rerank_screen SAMPLE OUTPUT.json

Builds the automatic index (`epzero.build`, one thread), reports what it chose, then searches it with
one thread at each width under three re-rank settings passed per search: the plan's (default), at least
ef candidates with the plan's early stop (deep), and exactly ef with no early stop (full). Aggregates only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

WIDTHS = (100, 150, 200, 300, 400, 600, 800, 1000)


def recall(found, exact, k):
    return float(np.mean([len(set(found[q, :k].tolist()) & set(exact[q, :k].tolist())) / k for q in range(len(exact))]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('sample', type=Path)
    parser.add_argument('output', type=Path)
    a = parser.parse_args()
    import epzero
    vectors = np.ascontiguousarray(np.load(a.sample / 'vectors.npy'))
    queries = np.ascontiguousarray(np.load(a.sample / 'queries.npy'))
    exact = np.load(a.sample / 'exact-top100.npy')
    metric = json.loads((a.sample / 'sample.json').read_text()).get('metric', 'ip')
    started = time.perf_counter()
    index = epzero.build(vectors, metric=metric, threads=1)
    build_seconds = time.perf_counter() - started
    plan = index.plan
    settings = {'default': lambda ef: (plan.rerank, plan.rerank_z),
                'deep': lambda ef: (max(plan.rerank, ef), plan.rerank_z),
                'full': lambda ef: (ef, 0.0)}
    sample = range(0, len(queries), max(1, len(queries) // 200))
    rows = []
    for width in WIDTHS:
        for k in (10, 100):
            for name, choose in settings.items():
                rerank, z = choose(width)
                kwargs = dict(k=k, ef=width, rerank=rerank, rerank_z=z, threads=1)
                index.search(queries[:20], **kwargs)                                  # warm
                started = time.perf_counter()
                labels, _ = index.search(queries, **kwargs)
                seconds = time.perf_counter() - started
                rows.append({'setting': name, 'ef': width, 'k': k, 'rerank': int(rerank), 'rerank_z': float(z),
                             'recall': recall(np.asarray(labels, dtype=np.int64), exact, k), 'qps': len(queries) / seconds})
    report = {'kind': 'EXPLORATORY development screen D12 (aggregates only)', 'sample': a.sample.name, 'metric': metric,
              'build_seconds': build_seconds,
              'plan': {f: getattr(plan, f) for f in ('quantization', 'nav_dim', 'rerank', 'rerank_z', 'refine', 'fill_after', 'ef')},
              'rows': rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'sample': report['sample'], 'plan': report['plan']}))


if __name__ == '__main__':
    main()
