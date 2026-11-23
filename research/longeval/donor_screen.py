"""Development screen D1: equal-degree donor intervention (design: docs/screen-d1-donor-intervention.md).

    python -m research.longeval.donor_screen SAMPLE OUTPUT.json

SAMPLE is an insertion-pilot sample (vectors, cohorts, lexical ranks, queries, exact top-100).
Builds one FAISS HNSW graph in arrival order, then, on copies of it, gives every snapshot-3 document
up to 4 new incoming base-layer links from its eligible exact neighbours, trying donors in a different
order per arm (closest, uniform, hub), and measures the arrival gap. Aggregates only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

from .insertion_pilot import BANDS, insertion_order
from .insertion_strata import risk_difference

PER_TARGET, DONOR_CAP, CANDIDATES = 4, 8, 32


def base_lists(index):
    import faiss
    hnsw = index.hnsw
    neighbors = faiss.vector_to_array(hnsw.neighbors)
    offsets = faiss.vector_to_array(hnsw.offsets)
    width = hnsw.nb_neighbors(0)
    rows = offsets[:-1].astype(np.int64)
    return neighbors, rows, width


def exact_neighbours(vectors, rows, k):
    """For each sample row in `rows`: its k most similar other rows (inner product), most similar first."""
    out = np.empty((len(rows), k), dtype=np.int64)
    for start in range(0, len(rows), 1024):
        block = rows[start:start + 1024]
        sims = np.asarray(vectors[block]) @ vectors.T
        sims[np.arange(len(block)), block] = -np.inf
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1, kind='stable')
        out[start:start + len(block)] = np.take_along_axis(top, order, axis=1)
    return out


def intervene(neighbors, rows, width, vectors, sample_of, targets, donors_of, arm, rng):
    """Edit base-layer lists in place. `targets` and donor IDs are storage IDs. Returns counts."""
    lists = neighbors  # flat array; node s occupies rows[s] .. rows[s] + width
    indegree = np.bincount(lists[lists >= 0], minlength=len(rows))
    accepted = np.zeros(len(rows), dtype=np.int64)
    added_edges = set()
    added = displaced = 0
    for t in targets:
        eligible = [d for d in donors_of[t] if t not in lists[rows[d]:rows[d] + width]]
        if arm == 'uniform':
            eligible = list(rng.permutation(eligible))
        elif arm == 'hub':
            eligible = sorted(eligible, key=lambda d: -indegree[d])          # stable: ties keep closeness order
        given = 0
        for d in eligible:
            if given == PER_TARGET:
                break
            if accepted[d] >= DONOR_CAP:
                continue
            slot = lists[rows[d]:rows[d] + width]
            free = np.flatnonzero(slot < 0)
            if len(free):
                j = free[0]
            else:
                current = slot.copy()
                sims = vectors[sample_of[current]] @ vectors[sample_of[d]]
                protected = np.array([(d, int(c)) in added_edges for c in current])
                sims[protected] = np.inf
                j = int(np.argmin(sims))
                if not np.isfinite(sims[j]):
                    continue
                displaced += 1
            lists[rows[d] + j] = t
            added_edges.add((d, int(t)))
            accepted[d] += 1
            given += 1
            added += 1
    return {'links_added': added, 'links_displaced': displaced}


def evaluate(index, sequence, queries, exact, cohort, widths=(100, 200, 400)):
    import faiss
    out = {}
    band_of = np.zeros(101, dtype=np.int64)
    for b, (lo, hi) in enumerate(BANDS):
        band_of[lo:hi + 1] = b
    for width in widths:
        index.hnsw.efSearch = width
        faiss.cvar.hnsw_stats.reset()
        started = time.perf_counter()
        _, labels = index.search(np.ascontiguousarray(queries), 100)
        seconds = time.perf_counter() - started
        found = sequence[labels]
        table = np.zeros((len(queries), 3, len(BANDS), 2), dtype=np.int64)
        for q in range(len(queries)):
            got = set(found[q].tolist())
            for rank, doc in enumerate(exact[q], 1):
                table[q, cohort[doc] - 1, band_of[rank], 0] += 1
                table[q, cohort[doc] - 1, band_of[rank], 1] += doc not in got
        out[str(width)] = {
            'newest_excess_pp': 100 * risk_difference(table, 3),
            'miss_rate_pct': {str(c): 100 * float(table[:, c - 1, :, 1].sum() / table[:, c - 1, :, 0].sum()) for c in (1, 2, 3)},
            'recall_at_100': float(1 - table[..., 1].sum() / table[..., 0].sum()),
            'distance_computations_per_query': faiss.cvar.hnsw_stats.ndis / len(queries),
            'ms_per_query': 1000 * seconds / len(queries)}
    return out


def main():
    import faiss
    faiss.omp_set_num_threads(1)
    sample, output = Path(sys.argv[1]), Path(sys.argv[2])
    vectors = np.load(sample / 'vectors.npy')
    cohort = np.load(sample / 'cohort.npy')
    queries = np.load(sample / 'queries.npy')
    exact = np.load(sample / 'exact-top100.npy')
    n, dim = vectors.shape
    sequence = insertion_order(np.load(sample / 'lexical-rank.npy'), cohort, 'appended', 11)   # storage -> sample
    position = np.empty(n, dtype=np.int64)
    position[sequence] = np.arange(n)                                                            # sample -> storage
    started = time.perf_counter()
    index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 100
    index.hnsw.rng = faiss.RandomGenerator(11)
    for row in sequence:
        index.add(vectors[row:row + 1])
    print('built', round(time.perf_counter() - started), 's', flush=True)
    frozen = faiss.serialize_index(index)
    report = {'kind': 'EXPLORATORY development screen D1 (aggregates only)', 'targets': int((cohort == 3).sum()),
              'per_target': PER_TARGET, 'donor_cap': DONOR_CAP, 'candidates': CANDIDATES,
              'arms': {'baseline': {'evaluation': evaluate(index, sequence, queries, exact, cohort)}}}
    print('baseline', json.dumps(report['arms']['baseline']['evaluation']['200']), flush=True)
    target_rows = np.flatnonzero(cohort == 3)
    started = time.perf_counter()
    near = exact_neighbours(vectors, target_rows, CANDIDATES)
    print('exact neighbours', round(time.perf_counter() - started), 's', flush=True)
    donors_of = {int(position[r]): [int(position[x]) for x in near[i]] for i, r in enumerate(target_rows)}
    order = np.random.default_rng(20261123).permutation(sorted(donors_of))
    neighbors, rows, width = base_lists(index)
    report['indegree_targets_before'] = float(np.bincount(neighbors[neighbors >= 0], minlength=n)[order].mean())
    del index
    for arm in ('closest', 'uniform', 'hub'):
        copy = faiss.deserialize_index(frozen)
        lists, rows, width = base_lists(copy)
        counts = intervene(lists, rows, width, vectors, sequence, order, donors_of, arm, np.random.default_rng(7))
        faiss.copy_array_to_vector(lists, copy.hnsw.neighbors)
        check, _, _ = base_lists(copy)
        if not np.array_equal(check, lists):
            raise RuntimeError('edited lists were not written back')
        counts['indegree_targets_after'] = float(np.bincount(lists[lists >= 0], minlength=n)[order].mean())
        report['arms'][arm] = {**counts, 'evaluation': evaluate(copy, sequence, queries, exact, cohort)}
        print(arm, json.dumps({k: v for k, v in report['arms'][arm].items() if k != 'evaluation'}),
              json.dumps(report['arms'][arm]['evaluation']['200']), flush=True)
        del copy
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
