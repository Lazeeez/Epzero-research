"""Development screen D17: a search width calibrated once, as the collection grows
(design: docs/screen-d17-calibration-staleness.md).

    python -m research.longeval.calibration_screen SAMPLE OUTPUT.json --arm grown --seed 11

Epzero's product path on the development sample, built snapshot by snapshot as in screens D13 and
D15. After each snapshot the search width reaching recall@100 0.95 on 100 calibration queries is found
again. The evaluation queries are scored at the width found after snapshot 1 (fixed) and at the
width found now (recalibrated). Aggregates only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .insertion_pilot import BANDS, insertion_order
from .insertion_strata import risk_difference

WIDTHS = (50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800)
TARGET = 0.95
SPLIT_SEED, CALIBRATION_QUERIES = 20261124, 100
ARMS = {'grown': {'extra_backlinks': 0, 'refine_recent_ef': None},
        'repaired8w': {'extra_backlinks': 8, 'refine_recent_ef': 100}}


def split(count):
    """(calibration rows, evaluation rows) of the test queries, fixed for every arm and seed."""
    order = np.random.default_rng(SPLIT_SEED).permutation(count)
    return np.sort(order[:CALIBRATION_QUERIES]), np.sort(order[CALIBRATION_QUERIES:])


def exact_among(vectors, queries, present, metric, k=100, chunk=20000):
    """Exact top-k sample positions among the `present` positions: float32 candidates, re-scored in
    float64, ties by position (as the sample's own exact neighbours)."""
    ids = np.flatnonzero(present)
    q32 = np.ascontiguousarray(queries, dtype=np.float32)
    keep = 2 * k
    best_ids = np.zeros((len(q32), 0), dtype=np.int64)
    best_scores = np.zeros((len(q32), 0), dtype=np.float32)
    for start in range(0, len(ids), chunk):
        part = ids[start:start + chunk]
        block = np.asarray(vectors[part], dtype=np.float32)
        scores = q32 @ block.T if metric == 'ip' else -((q32 ** 2).sum(1)[:, None] - 2 * q32 @ block.T + (block ** 2).sum(1)[None])
        scores = np.concatenate([best_scores, scores], axis=1)
        pool = np.concatenate([best_ids, np.broadcast_to(part, (len(q32), len(part)))], axis=1)
        top = np.argpartition(-scores, min(keep, scores.shape[1] - 1), axis=1)[:, :keep]
        best_scores, best_ids = np.take_along_axis(scores, top, 1), np.take_along_axis(pool, top, 1)
    out = np.empty((len(q32), k), dtype=np.int64)
    q64 = np.asarray(queries, dtype=np.float64)
    for qi in range(len(q64)):
        cand = best_ids[qi]
        v = np.asarray(vectors[np.sort(cand)], dtype=np.float64)
        cand = np.sort(cand)
        s = v @ q64[qi] if metric == 'ip' else -((v - q64[qi]) ** 2).sum(1)
        out[qi] = cand[np.lexsort((cand, -s))[:k]]
    return out


def table_of(found, targets, cohort):
    band_of = np.zeros(101, dtype=np.int64)
    for b, (lo, hi) in enumerate(BANDS):
        band_of[lo:hi + 1] = b
    table = np.zeros((len(targets), 3, len(BANDS), 2), dtype=np.int64)
    for q in range(len(targets)):
        got = set(found[q].tolist())
        for rank, doc in enumerate(targets[q], 1):
            table[q, cohort[doc] - 1, band_of[rank]] += (1, doc not in got)
    return table


def summary(table, newest):
    out = {'recall_at_100': float(1 - table[..., 1].sum() / table[..., 0].sum()),
           'miss_rate_pct': {str(c): 100 * float(table[:, c - 1, :, 1].sum() / table[:, c - 1, :, 0].sum())
                             for c in range(1, newest + 1)}}
    if newest > 1:
        out['newest_excess_pp'] = 100 * risk_difference(table, newest)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('sample', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--arm', choices=sorted(ARMS), required=True)
    parser.add_argument('--seed', type=int, required=True)
    a = parser.parse_args()
    import epzero
    setting = ARMS[a.arm]
    vectors = np.load(a.sample / 'vectors.npy', mmap_mode='r')
    cohort = np.load(a.sample / 'cohort.npy')
    queries = np.ascontiguousarray(np.load(a.sample / 'queries.npy'))
    full_exact = np.load(a.sample / 'exact-top100.npy')
    metric = json.loads((a.sample / 'sample.json').read_text()).get('metric', 'ip')
    n, dim = vectors.shape
    calibration, evaluation = split(len(queries))
    sequence = insertion_order(np.load(a.sample / 'lexical-rank.npy'), cohort, 'appended', a.seed)
    index = epzero.Index(dim=dim, metric=metric, max_elements=n, M=16, ef_construction=100, seed=a.seed,
                         quantization='sq8', keep_vectors=True, nav_dim=0, fill_links=False, fast_build=True,
                         half_vectors=False, extra_backlinks=setting['extra_backlinks'])
    first = np.sort(sequence[cohort[sequence] == 1])
    bulk = np.ascontiguousarray(vectors[first])
    index.train(bulk)
    index.add(bulk, labels=first.astype(np.int64), threads=0)
    del bulk
    index.refine_links(threads=0, ef=50)
    snapshots, fixed = {}, None
    for step in (1, 2, 3):
        started = time.perf_counter()
        if step > 1:
            batch = sequence[cohort[sequence] == step]
            position = len(index)
            for start in range(0, len(batch), 1024):
                chunk = batch[start:start + 1024]
                index.add(np.ascontiguousarray(vectors[chunk]), labels=chunk.astype(np.int64), threads=1)
            if setting['refine_recent_ef']:
                index.refine_links(threads=0, ef=setting['refine_recent_ef'], first=position)
        build_seconds = time.perf_counter() - started
        targets = exact_among(vectors, queries, cohort <= step, metric)
        check = None
        if step == 3:
            check = float(np.mean([len(set(targets[q].tolist()) & set(full_exact[q].tolist())) / 100 for q in range(len(queries))]))
            if check < 0.999:
                raise ValueError(f'exact neighbours of the whole sample disagree with the sample: overlap {check}')
        found = {w: np.asarray(index.search(queries, k=100, ef=w, rerank=w, threads=1)[0], dtype=np.int64) for w in WIDTHS}
        cal_recall = {w: float(np.mean([len(set(found[w][q].tolist()) & set(targets[q].tolist())) / 100 for q in calibration]))
                      for w in WIDTHS}
        reached = [w for w in WIDTHS if cal_recall[w] >= TARGET]
        width = reached[0] if reached else WIDTHS[-1]
        if fixed is None:
            fixed = width
        snapshots[str(step)] = {
            'documents': int((cohort <= step).sum()), 'build_seconds': build_seconds,
            'recalibrated_width': width, 'target_reached': bool(reached), 'fixed_width': fixed,
            'calibration_recall_by_width': {str(w): cal_recall[w] for w in WIDTHS},
            'evaluation_fixed': summary(table_of(found[fixed][evaluation], targets[evaluation], cohort), step),
            'evaluation_recalibrated': summary(table_of(found[width][evaluation], targets[evaluation], cohort), step),
            'exact_overlap_with_sample': check}
        print(json.dumps({'step': step, 'fixed': fixed, 'recalibrated': width,
                          'recall_fixed': snapshots[str(step)]['evaluation_fixed']['recall_at_100']}), flush=True)
    report = {'kind': 'EXPLORATORY development screen D17 (aggregates only)', 'arm': a.arm, 'setting': setting, 'seed': a.seed,
              'metric': metric, 'target': TARGET, 'calibration_queries': int(len(calibration)),
              'evaluation_queries': int(len(evaluation)), 'widths': list(WIDTHS), 'snapshots': snapshots}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
