"""Screen M1: a model of passive accumulation, and how the disadvantage ages (design:
docs/screen-m1-accumulation-model.md, recorded before this code was written).

    python -m research.longeval.accumulation run SAMPLE OUTPUT --order shuffled --seed 11
    python -m research.longeval.accumulation analyze OUTPUT.json BUILD [BUILD ...]

`run` builds one Epzero index one vector per call (uncompressed, M 16, efConstruction 100, one thread),
reads each node's layer-0 list right after its own insertion (its choices), and at checkpoints after
every tenth of the insertions records every node's in-degree split into links from later and from
earlier nodes, and the queries' misses at ef 200 against the exact top 100 of the nodes inserted so far.
`analyze` computes the screen's measures. Aggregates only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .insertion_pilot import BANDS, insertion_order
from .insertion_strata import mantel_haenszel

CHECKPOINTS = 10
EF, K, CANDIDATES = 200, 100, 300
BINS = 100


def exact_prefix(vectors, queries, inserted, metric, *, chunk=50_000):
    """Exact top-K over the rows `inserted` (sample positions): float32 scan for CANDIDATES, re-scored
    in float64, best first, ties by ascending sample position."""
    q32 = np.ascontiguousarray(queries, dtype=np.float32)
    q64 = queries.astype(np.float64)
    order = np.sort(inserted)
    best_ids = np.empty((len(queries), 0), dtype=np.int64)
    best_key = np.empty((len(queries), 0), dtype=np.float32)
    for start in range(0, len(order), chunk):
        ids = order[start:start + chunk]
        block = np.asarray(vectors[ids], dtype=np.float32)
        if metric == 'ip':
            key = -(q32 @ block.T)
        else:
            key = (block * block).sum(axis=1)[None, :] - 2 * (q32 @ block.T)
        key = np.concatenate([best_key, key], axis=1)
        cand = np.concatenate([np.broadcast_to(best_ids, (len(queries), best_ids.shape[1])),
                               np.broadcast_to(ids, (len(queries), len(ids)))], axis=1)
        keep = np.argpartition(key, min(CANDIDATES, key.shape[1] - 1), axis=1)[:, :CANDIDATES]
        best_key = np.take_along_axis(key, keep, axis=1)
        best_ids = np.take_along_axis(cand, keep, axis=1)
    top = np.empty((len(queries), K), dtype=np.int64)
    for qi in range(len(queries)):
        ids = np.sort(best_ids[qi])
        rows = np.asarray(vectors[ids], dtype=np.float64)
        score = -(rows @ q64[qi]) if metric == 'ip' else ((rows - q64[qi]) ** 2).sum(axis=1)
        top[qi] = ids[np.argsort(score, kind='stable')[:K]]
    return top


def miss_table(found, targets, group, groups):
    """[query, group, band, (n, missed)] for one search."""
    band_of = np.zeros(K + 1, dtype=np.int64)
    for b, (lo, hi) in enumerate(BANDS):
        band_of[lo:hi + 1] = b
    table = np.zeros((len(targets), groups, len(BANDS), 2), dtype=np.int64)
    ranks = band_of[np.arange(1, K + 1)]
    for qi in range(len(targets)):
        missed = ~np.isin(targets[qi], found[qi])
        g = group[targets[qi]]
        np.add.at(table[qi], (g, ranks, 0), 1)
        np.add.at(table[qi], (g, ranks, 1), missed.astype(np.int64))
    return table


def run(sample, output, *, order, seed, M=16, ef_construction=100):
    import epzero
    sample, output = Path(sample), Path(output)
    vectors = np.load(sample / 'vectors.npy', mmap_mode='r')
    cohort = np.load(sample / 'cohort.npy')
    queries = np.load(sample / 'queries.npy')
    metric = json.loads((sample / 'sample.json').read_text()).get('metric', 'ip')
    n, dim = vectors.shape
    sequence = insertion_order(np.load(sample / 'lexical-rank.npy'), cohort, order, seed)
    position = np.empty(n, dtype=np.int64)
    position[sequence] = np.arange(n)
    index = epzero.Index(dim=dim, metric=metric, max_elements=n, M=M, ef_construction=ef_construction, seed=seed,
                         quantization='none', keep_vectors=True, nav_dim=0, fill_links=False,
                         fast_build=False, half_vectors=False)
    choices = np.zeros(n, dtype=np.int64)     # k_j, by insertion position
    chosen = np.zeros(n, dtype=np.int64)      # C_i, by sample position
    marks = [n * c // CHECKPOINTS for c in range(1, CHECKPOINTS + 1)]
    output.mkdir(parents=True, exist_ok=False)
    checkpoints, started = [], time.perf_counter()
    for p, label in enumerate(sequence):
        index.add(np.ascontiguousarray(vectors[label:label + 1]), labels=np.array([label], dtype=np.int64), threads=1)
        links = np.asarray(index.neighbors(int(label), 0), dtype=np.int64)
        choices[p] = len(links)
        np.add.at(chosen, links, 1)
        if p + 1 in marks:
            t = p + 1
            inserted = sequence[:t]
            later = np.zeros(n, dtype=np.int64)
            earlier = np.zeros(n, dtype=np.int64)
            for source in inserted:
                targets = np.asarray(index.neighbors(int(source), 0), dtype=np.int64)
                newer = position[source] > position[targets]
                np.add.at(later, targets[newer], 1)
                np.add.at(earlier, targets[~newer], 1)
            exact = exact_prefix(vectors, queries, inserted, metric)
            found, _ = index.search(np.ascontiguousarray(queries), k=K, ef=EF, rerank=0, rerank_z=0, patience=0, threads=1)
            found = np.asarray(found, dtype=np.int64)
            local = np.full(n, 0, dtype=np.int64)
            local[inserted] = np.arange(t) * 10 // t                    # tenth of the prefix
            fixed = position * 10 // n                                   # tenth of the full build
            c = len(checkpoints) + 1
            np.save(output / f'prefix-table-{c:02d}.npy', miss_table(found, exact, local, 10))
            np.save(output / f'fixed-table-{c:02d}.npy', miss_table(found, exact, fixed, 10))
            np.save(output / f'later-{c:02d}.npy', later[sequence[:t]].astype(np.int32))       # by insertion position
            np.save(output / f'earlier-{c:02d}.npy', earlier[sequence[:t]].astype(np.int32))
            recall = float(np.mean([np.isin(exact[q], found[q]).mean() for q in range(len(queries))]))
            checkpoints.append({'inserted': t, 'recall_at_100': recall, 'seconds': time.perf_counter() - started})
            print(json.dumps({'order': order, 'seed': seed, **checkpoints[-1]}), flush=True)
    np.save(output / 'choices.npy', choices.astype(np.int32))
    np.save(output / 'chosen.npy', chosen[sequence].astype(np.int32))                         # by insertion position
    report = {'sample': str(sample.name), 'metric': metric, 'count': int(n), 'order': order, 'seed': seed, 'M': M,
              'ef_construction': ef_construction, 'ef': EF, 'k': K, 'epzero': epzero.__version__,
              'queries': int(len(queries)), 'checkpoints': checkpoints}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def exposure(choices, upto=None):
    """E_p = sum over later positions q < upto of k_q / q (0-based positions; q nodes present when q arrives)."""
    n = len(choices) if upto is None else upto
    k = choices[:n].astype(np.float64)
    term = np.zeros(n)
    term[1:] = k[1:] / np.arange(1, n)
    tail = np.cumsum(term[::-1])[::-1]               # sum over q >= p
    return np.concatenate([tail[1:], [0.0]])         # sum over q > p


def tenths(values):
    n = len(values)
    return [float(values[d * n // 10:(d + 1) * n // 10].mean()) for d in range(10)]


def measures(build):
    build = Path(build)
    report = json.loads((build / 'report.json').read_text())
    n = report['count']
    choices = np.load(build / 'choices.npy').astype(np.int64)
    chosen = np.load(build / 'chosen.npy').astype(np.int64)
    E = exposure(choices)
    later = np.load(build / f'later-{CHECKPOINTS:02d}.npy').astype(np.float64)
    earlier = np.load(build / f'earlier-{CHECKPOINTS:02d}.npy').astype(np.float64)
    C_t, E_t = tenths(chosen.astype(np.float64)), tenths(E)
    # Shape: S against ln(n / i) over percentile bins, the first excluded (1-based positions).
    edges = [n * b // BINS for b in range(BINS + 1)]
    x = np.array([np.log(n / np.arange(edges[b] + 1, edges[b + 1] + 1)).mean() for b in range(1, BINS)])
    y = np.array([later[edges[b]:edges[b + 1]].mean() for b in range(1, BINS)])
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (slope * x + intercept)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    S_t, R_t = tenths(later), tenths(earlier)
    # Aging in the graph: nodes in the first and fifth tenths of the first half, checkpoints 50%..100%.
    first = slice(0, n // 20)
    fifth = slice(n // 5, n // 4)
    gaps = []
    for c in range(5, CHECKPOINTS + 1):
        degree = (np.load(build / f'later-{c:02d}.npy').astype(np.float64)
                  + np.load(build / f'earlier-{c:02d}.npy').astype(np.float64))
        gaps.append(float(degree[first].mean() - degree[fifth].mean()))
    # Aging in search.
    newest, fixed5 = [], {}
    for c in range(1, CHECKPOINTS + 1):
        prefix = np.load(build / f'prefix-table-{c:02d}.npy')
        newest.append(mantel_haenszel(prefix, 10, 1))
        if c >= 5:
            fixed5[str(10 * c)] = mantel_haenszel(np.load(build / f'fixed-table-{c:02d}.npy'), 5, 1)
    return {'order': report['order'], 'seed': report['seed'], 'sample': report['sample'],
            'recall_by_checkpoint': [cp['recall_at_100'] for cp in report['checkpoints']],
            'mean_choices': float(choices[1:].mean()),
            'chosen_by_tenth': C_t, 'exposure_by_tenth': E_t,
            'exposure_share_of_chosen_gradient': float((E_t[0] - E_t[-1]) / (C_t[0] - C_t[-1])),
            'chosen_over_exposure_by_tenth': [c / e if e else None for c, e in zip(C_t, E_t)],
            'later_by_tenth': S_t, 'earlier_by_tenth': R_t,
            'surviving_share_of_choices_by_tenth': [s / c if c else None for s, c in zip(S_t, C_t)],
            'later_vs_log_fit': {'slope': float(slope), 'intercept': float(intercept), 'r2': float(r2)},
            'earlier_gradient_over_later_gradient': float((R_t[0] - R_t[-1]) / (S_t[0] - S_t[-1])),
            'degree_gap_first_vs_fifth_twentieth': {str(10 * c): g for c, g in zip(range(5, CHECKPOINTS + 1), gaps)},
            'degree_gap_relative_change_50_to_100': float((gaps[-1] - gaps[0]) / gaps[0]),
            'newest_tenth_vs_first_ratio_by_checkpoint': {str(10 * c): v for c, v in zip(range(1, CHECKPOINTS + 1), newest)},
            'fixed_fifth_tenth_vs_first_ratio_by_checkpoint': fixed5}


def analyze(builds):
    per = [measures(b) for b in builds]
    out = {'kind': 'EXPLORATORY screen M1 (docs/screen-m1-accumulation-model.md), aggregates only', 'builds': per,
           'predictions': {}}
    for m in per:
        key = f"{m['sample']}-{m['order']}-{m['seed']}"
        from_30 = [v for c, v in m['newest_tenth_vs_first_ratio_by_checkpoint'].items() if int(c) >= 30]
        fixed = m['fixed_fifth_tenth_vs_first_ratio_by_checkpoint']
        out['predictions'][key] = {
            '1_exposure_explains_at_least_80pct': m['exposure_share_of_chosen_gradient'] >= 0.8,
            '2_later_linear_in_log_r2_at_least_0.95': m['later_vs_log_fit']['r2'] >= 0.95,
            '2_earlier_gradient_below_a_third': abs(m['earlier_gradient_over_later_gradient']) < 1 / 3,
            '3_degree_gap_changes_below_20pct': abs(m['degree_gap_relative_change_50_to_100']) < 0.2,
            '4_newest_ratio_above_1.2_from_30pct': all(v > 1.2 for v in from_30),
            '4_fixed_group_keeps_over_half': (fixed['100'] - 1) > 0.5 * (fixed['50'] - 1)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('sample', type=Path)
    r.add_argument('output', type=Path)
    r.add_argument('--order', choices=('appended', 'shuffled'), required=True)
    r.add_argument('--seed', type=int, required=True)
    r.add_argument('--M', type=int, default=16)
    r.add_argument('--ef-construction', type=int, default=100)
    a = sub.add_parser('analyze')
    a.add_argument('output', type=Path)
    a.add_argument('builds', type=Path, nargs='+')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.sample, args.output, order=args.order, seed=args.seed, M=args.M, ef_construction=args.ef_construction)
    else:
        out = analyze(args.builds)
        args.output.write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps(out['predictions'], indent=2))


if __name__ == '__main__':
    main()
