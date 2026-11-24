"""The repaired-Epzero arm at full scale (design and predictions: scripts/pace/LONGEVAL_ENGINE_ARM.md).

    python -m research.longeval.engine_arm SAMPLE OUTPUT --arm repaired --seed 11

Epzero's product path on all of snapshot 3, as the collection grew: the snapshot-1 documents are
trained on and bulk-built with 8-bit codes on all cores, then refined; the snapshot-2 and snapshot-3
documents are inserted one at a time, in arrival order. Arms:

  grown     no repair (the current behaviour)
  repaired  `extra_backlinks` 4, and after each later snapshot the documents it added are refined
  repaired8 the same with `extra_backlinks` 8 (screen D13; second run, package -02)
  rebuilt   every document trained on, bulk-built and refined at once (reference without history)

Needs the Epzero build with `extra_backlinks` and `refine_links(first=...)`. SAMPLE is a full-scale
insertion-pilot sample (`insertion_pilot prepare-full`, with the confirmation queries). Both query
sets are searched at ef 100, 200 and 400 (k 100, re-ranking the whole beam with full vectors); per
query tables are saved for a query bootstrap, and returned IDs for paired analyses. Aggregates and
positions only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .insertion_pilot import BANDS, insertion_order

WIDTHS = (100, 200, 400)
ARMS = {'grown': {'extra_backlinks': 0, 'refine_recent': False, 'rebuild': False},
        'repaired': {'extra_backlinks': 4, 'refine_recent': True, 'rebuild': False},
        'repaired8': {'extra_backlinks': 8, 'refine_recent': True, 'rebuild': False},
        'rebuilt': {'extra_backlinks': 0, 'refine_recent': False, 'rebuild': True}}
M, EF_CONSTRUCTION, REFINE_EF = 16, 100, 50        # REFINE_EF: what epzero.build() uses, max(efC / 2, 2M)


def build(vectors, cohort, sequence, *, arm, seed, metric):
    import epzero
    setting = ARMS[arm]
    n, dim = vectors.shape
    index = epzero.Index(dim=dim, metric=metric, max_elements=n, M=M, ef_construction=EF_CONSTRUCTION, seed=seed,
                         quantization='sq8', keep_vectors=True, nav_dim=0, fill_links=False, fast_build=True,
                         half_vectors=False, extra_backlinks=setting['extra_backlinks'])
    first = sequence if setting['rebuild'] else sequence[cohort[sequence] == 1]
    later = sequence[:0] if setting['rebuild'] else sequence[cohort[sequence] != 1]
    timing = {}
    started = time.perf_counter()
    bulk = np.ascontiguousarray(vectors[np.sort(first)])
    index.train(bulk)
    index.add(bulk, labels=np.sort(first).astype(np.int64), threads=0)
    del bulk
    index.refine_links(threads=0, ef=REFINE_EF)
    timing['bulk_seconds'] = time.perf_counter() - started
    timing['insert_seconds'] = timing['refine_recent_seconds'] = 0.0
    for step in (2, 3):
        batch = later[cohort[later] == step]
        if not len(batch):
            continue
        position = len(index)                          # nothing is removed
        started = time.perf_counter()
        for start in range(0, len(batch), 1024):
            chunk = batch[start:start + 1024]
            index.add(np.ascontiguousarray(vectors[chunk]), labels=chunk.astype(np.int64), threads=1)
        timing['insert_seconds'] += time.perf_counter() - started
        if setting['refine_recent']:
            started = time.perf_counter()
            index.refine_links(threads=0, ef=REFINE_EF, first=position)
            timing['refine_recent_seconds'] += time.perf_counter() - started
    return index, timing


def evaluate(index, queries, exact, cohort, prefix, output):
    band_of = np.zeros(101, dtype=np.int64)
    for b, (lo, hi) in enumerate(BANDS):
        band_of[lo:hi + 1] = b
    out = {}
    for width in WIDTHS:
        started = time.perf_counter()
        labels, _ = index.search(queries, k=100, ef=width, rerank=width, threads=1)
        seconds = time.perf_counter() - started
        found = np.asarray(labels, dtype=np.int64)
        table = np.zeros((len(queries), 3, len(BANDS), 2), dtype=np.int64)
        for q in range(len(queries)):
            got = set(found[q].tolist())
            for rank, doc in enumerate(exact[q], 1):
                table[q, cohort[doc] - 1, band_of[rank], 0] += 1
                table[q, cohort[doc] - 1, band_of[rank], 1] += doc not in got
        recall10 = [len(set(found[q, :10].tolist()) & set(exact[q, :10].tolist())) / 10 for q in range(len(queries))]
        work = [index.trace(queries[q:q + 1], k=100, ef=width)['distance_computations'] for q in range(len(queries))]
        np.save(output / f'{prefix}per-query-ef{width}.npy', table)
        np.save(output / f'{prefix}labels-ef{width}.npy', found.astype(np.int32))
        np.save(output / f'{prefix}recall10-ef{width}.npy', np.asarray(recall10))
        np.save(output / f'{prefix}work-ef{width}.npy', np.asarray(work, dtype=np.int64))
        out[str(width)] = {'miss_rate_pct': {str(c): 100 * float(table[:, c - 1, :, 1].sum() / table[:, c - 1, :, 0].sum())
                                             for c in (1, 2, 3)},
                           'recall_at_100': float(1 - table[..., 1].sum() / table[..., 0].sum()),
                           'recall_at_10_top100_list': float(np.mean(recall10)),
                           'distance_computations_per_query': float(np.mean(work)),
                           'ms_per_query': 1000 * seconds / len(queries)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('sample', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--arm', choices=sorted(ARMS), required=True)
    parser.add_argument('--seed', type=int, required=True)
    a = parser.parse_args()
    vectors = np.load(a.sample / 'vectors.npy', mmap_mode='r')
    cohort = np.load(a.sample / 'cohort.npy')
    metric = json.loads((a.sample / 'sample.json').read_text()).get('metric', 'ip')
    sequence = insertion_order(np.load(a.sample / 'lexical-rank.npy'), cohort, 'appended', a.seed)
    index, timing = build(vectors, cohort, sequence, arm=a.arm, seed=a.seed, metric=metric)
    out = a.output / f'{a.arm}-{a.seed}'
    out.mkdir(parents=True, exist_ok=False)
    report = {'arm': a.arm, 'seed': a.seed, 'setting': ARMS[a.arm], 'M': M, 'ef_construction': EF_CONSTRUCTION,
              'refine_ef': REFINE_EF, 'metric': metric, 'documents': int(len(cohort)), **timing,
              'test': evaluate(index, np.ascontiguousarray(np.load(a.sample / 'queries.npy')),
                               np.load(a.sample / 'exact-top100.npy'), cohort, '', out)}
    if (a.sample / 'confirm-queries.npy').exists():
        report['confirm'] = evaluate(index, np.ascontiguousarray(np.load(a.sample / 'confirm-queries.npy')),
                                     np.load(a.sample / 'confirm-exact-top100.npy'), cohort, 'confirm-', out)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('arm', 'seed', 'bulk_seconds', 'insert_seconds', 'refine_recent_seconds')}))


if __name__ == '__main__':
    main()
