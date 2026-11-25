"""Analysis R1: three reanalyses of existing outputs (design: docs/analysis-r1-reanalyses.md, recorded
before any of them was computed).

    python -m research.longeval.reanalyses margin INPUTS RELEASE FROZEN_EVIDENCE EXTENSION_EVIDENCE OUT.json
    python -m research.longeval.reanalyses resample SEEDS_EVIDENCE OUT.json
    python -m research.longeval.reanalyses fixed-effects OUT.json BACKEND=ORDER_FOLDER [BACKEND=ORDER_FOLDER ...]

1. `margin`: the protocol's calibration rule with targets 0.95 (check), 0.96 and 0.97, applied to the
   frozen held-out reports' calibration points; the test rankings at the chosen widths come from the
   held-out extension's sweep of the same incremental indexes.
2. `resample`: how often the 0.95 rule keeps ef 200 or less at snapshot 3 when the calibration set is
   drawn from the 50 calibration queries (the seven further seeds' per-query records).
3. `fixed-effects`: a binomial logistic model of misses on arrival group and exact-rank band with a
   fixed effect per query, fitted to the full-collection order rerun's per-query count tables.
Aggregates only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import read_run
from .heldout import ROLE, load_heldout
from .heldout_analysis import load_test_qrels, panel
from .heldout_sensitivity import FAMILY, ideal_dcg, ndcg10, ranked
from .insertion_strata import tables as order_tables

SEEDS = (11, 29, 47)
GRID = (100, 200, 400, 800)
TARGETS = (0.95, 0.96, 0.97)
DRAWS, RNG = 10_000, 20261126
FROZEN_D = {'epzero': -0.0059379, 'faiss_hnsw': -0.0007}


def choose(points, target):
    """The smallest grid width whose mean calibration recall reaches the target, else the largest."""
    recall = {p['effort']: p['recall'] for p in points}
    for width in GRID:
        if recall[width] >= target:
            return width
    return GRID[-1]


def _frozen_reports(evidence):
    out = {}
    for path in Path(evidence).rglob('trajectory/report.json'):
        r = json.loads(path.read_text())
        if r.get('mode') == 'incremental' and r['backend'] in ('epzero', 'faiss_hnsw'):
            out[(r['backend'], r['seed'])] = r
    return out


def _sweeps(evidence):
    out = {}
    for job in Path(evidence).glob('**/jobs/*'):
        arm = dict(item.split('=', 1) for item in (job / 'arm.txt').read_text().split())
        out[(arm['backend'], int(arm['seed']))] = job / 'incremental' / 'sweep'
    return out


def _bootstrap(values, draws=DRAWS, seed=RNG):
    rows = np.random.default_rng(seed).integers(0, len(values), (draws, len(values)))
    return np.quantile(values[rows].mean(axis=1), [.025, .975]).tolist()


def margin(inputs, release, frozen_evidence, extension_evidence):
    queries, refs, _, _ = load_heldout(inputs)
    ids = queries[ROLE].ids
    exact = {s: {q: refs[s - 1][q] for q in ids} for s in (1, 2, 3)}
    qrels = load_test_qrels(release)[0][FAMILY]
    population = panel(qrels, ids)
    reports, sweeps = _frozen_reports(frozen_evidence), _sweeps(extension_evidence)
    cache = {}

    def run(backend, seed, step, width):
        key = (backend, seed, step, width)
        if key not in cache:
            cache[key] = read_run(sweeps[(backend, seed)] / f'snapshot-{step}' / f'ef-{width}.run')
        return cache[key]

    def gap(backend, seed, step, width):
        r = run(backend, seed, step, width)
        out = []
        for q in population:
            grades = qrels[step].get(q, {})
            ideal = ideal_dcg(grades)
            out.append(ndcg10(ranked(r[q]), grades, ideal) - ndcg10(ranked(exact[step][q]), grades, ideal))
        return np.array(out)

    def recall(backend, seed, step, width):
        r = run(backend, seed, step, width)
        return float(np.mean([len({d for d, _ in r[q]} & {d for d, _ in exact[step][q][:100]}) / 100 for q in ids]))

    out = {'kind': 'EXPLORATORY analysis R1.1: calibration margin (aggregates only)', 'panel': len(population),
           'seeds': list(SEEDS), 'grid': list(GRID), 'arms': {}}
    for backend in ('epzero', 'faiss_hnsw'):
        block = {}
        for target in TARGETS:
            widths = {s: [choose(reports[(backend, s)]['snapshots'][i]['calibration']['points'], target) for i in range(3)]
                      for s in SEEDS}
            d = np.mean([gap(backend, s, 3, widths[s][2]) - gap(backend, s, 1, widths[s][0]) for s in SEEDS], axis=0)
            gaps = {str(step): float(np.mean([gap(backend, s, step, widths[s][step - 1]).mean() for s in SEEDS]))
                    for step in (1, 2, 3)}
            block[f'{target:.2f}'] = {
                'widths_by_seed': {str(s): w for s, w in widths.items()},
                'gap_by_snapshot': gaps, 'D': float(d.mean()), 'D_interval_95': _bootstrap(d),
                'D_within_0.01': all(abs(x) <= 0.01 for x in _bootstrap(d)),
                'test_recall_by_snapshot': {str(step): float(np.mean([recall(backend, s, step, widths[s][step - 1]) for s in SEEDS]))
                                            for step in (1, 2, 3)},
                'mean_width': float(np.mean([w for ws in widths.values() for w in ws]))}
        block['check_0.95_reproduces_frozen_D'] = abs(block['0.95']['D'] - FROZEN_D[backend]) < 5e-5
        out['arms'][backend] = block
    return out


def resample(seeds_evidence, sizes=(25, 50, 100, 200, 400), draws=5000):
    reports = {}
    for path in Path(seeds_evidence).rglob('report.json'):
        r = json.loads(path.read_text())
        if r.get('backend') in ('epzero', 'faiss_hnsw') and r.get('mode') == 'incremental':
            reports[(r['backend'], r['seed'])] = r
    rng = np.random.default_rng(RNG)
    out = {'kind': 'EXPLORATORY analysis R1.2: calibration-size study resampling the calibration queries (aggregates only)',
           'snapshot': 3, 'target': 0.95, 'draws': draws, 'note': 'sizes above 50 resample a 50-query pool', 'arms': {}}
    for backend in ('epzero', 'faiss_hnsw'):
        seeds = sorted(s for b, s in reports if b == backend)
        keep = {m: [] for m in sizes}
        for seed in seeds:
            per = reports[(backend, seed)]['snapshots'][2]['diagnostic']['calibration_recall_per_query']
            names = sorted(per[str(GRID[0])])
            matrix = np.array([[per[str(w)][q] for w in GRID] for q in names])      # [query, width]
            for m in sizes:
                rows = rng.integers(0, len(names), (draws, m))
                means = matrix[rows].mean(axis=1)                                     # [draw, width]
                reached = means >= 0.95
                chosen = np.where(reached.any(axis=1), np.argmax(reached, axis=1), len(GRID) - 1)
                keep[m].append(float((np.array(GRID)[chosen] <= 200).mean()))
        out['arms'][backend] = {'seeds': seeds, 'keeps_ef200_or_less': {str(m): float(np.mean(v)) for m, v in keep.items()},
                                'by_seed': {str(m): v for m, v in keep.items()}}
    return out


def fit_fixed_effects(table, iterations=50):
    """Binomial logit: miss ~ query + arrival group (2, 3 vs 1) + band (2, 3 vs 1), from [query, group, band, (n, missed)].
    Queries with no miss or no hit are dropped. Returns the log odds ratios of groups 2 and 3."""
    n, y = table[..., 0].astype(np.float64), table[..., 1].astype(np.float64)
    keep = (y.sum(axis=(1, 2)) > 0) & ((n - y).sum(axis=(1, 2)) > 0)
    n, y = n[keep], y[keep]
    queries, groups, bands = n.shape
    cells = [(q, g, b) for q in range(queries) for g in range(groups) for b in range(bands) if n[q, g, b] > 0]
    qi = np.array([c[0] for c in cells])
    X = np.zeros((len(cells), queries + (groups - 1) + (bands - 1)))
    X[np.arange(len(cells)), qi] = 1
    for i, (_, g, b) in enumerate(cells):
        if g:
            X[i, queries + g - 1] = 1
        if b:
            X[i, queries + groups - 1 + b - 1] = 1
    N = np.array([n[c] for c in cells])
    Y = np.array([y[c] for c in cells])
    beta = np.zeros(X.shape[1])
    base = np.log((y.sum(axis=(1, 2)) + 0.5) / (n.sum(axis=(1, 2)) - y.sum(axis=(1, 2)) + 0.5))
    beta[:queries] = base
    for _ in range(iterations):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = N * p * (1 - p)
        grad = X.T @ (Y - N * p)
        step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1]), grad)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta[queries:queries + groups - 1], int(keep.sum())


def fixed_effects(folders, ef=200, draws=5000):
    rng = np.random.default_rng(RNG)
    out = {'kind': 'EXPLORATORY analysis R1.3: fixed-effects logistic check of the within-query ratio (aggregates only)',
           'ef': ef, 'bootstrap': {'draws': draws, 'unit': 'query, seeds summed within query'}, 'backends': {}}
    for backend, folder in folders.items():
        found = order_tables(folder, ef=ef)
        block = {}
        for order in ('appended', 'shuffled', 'reversed'):
            table = np.sum([t for (o, _), t in found.items() if o == order], axis=0)
            coef, used = fit_fixed_effects(table)
            boot = []
            for _ in range(draws):
                rows = rng.integers(0, table.shape[0], table.shape[0])
                boot.append(fit_fixed_effects(table[rows])[0][1])
            block[order] = {'odds_ratio_snapshot3_vs_1': float(np.exp(coef[1])), 'odds_ratio_snapshot2_vs_1': float(np.exp(coef[0])),
                            'interval_95_snapshot3_vs_1': np.exp(np.quantile(boot, [.025, .975])).tolist(), 'queries_used': used}
        out['backends'][backend] = block
    out['prediction'] = {b: {'orders_as_mh': blk['appended']['odds_ratio_snapshot3_vs_1'] > 1 > blk['reversed']['odds_ratio_snapshot3_vs_1']
                             and blk['appended']['odds_ratio_snapshot3_vs_1'] > blk['shuffled']['odds_ratio_snapshot3_vs_1'] > blk['reversed']['odds_ratio_snapshot3_vs_1'],
                             'as_arrived_excludes_1': blk['appended']['interval_95_snapshot3_vs_1'][0] > 1}
                         for b, blk in out['backends'].items()}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('margin')
    for name in ('inputs', 'release', 'frozen_evidence', 'extension_evidence', 'output'):
        m.add_argument(name, type=Path)
    r = sub.add_parser('resample')
    r.add_argument('seeds_evidence', type=Path)
    r.add_argument('output', type=Path)
    f = sub.add_parser('fixed-effects')
    f.add_argument('output', type=Path)
    f.add_argument('folders', nargs='+', help='BACKEND=FOLDER of the order rerun')
    args = parser.parse_args()
    if args.command == 'margin':
        out = margin(args.inputs, args.release, args.frozen_evidence, args.extension_evidence)
    elif args.command == 'resample':
        out = resample(args.seeds_evidence)
    else:
        out = fixed_effects(dict(item.split('=', 1) for item in args.folders))
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2)[:5000])


if __name__ == '__main__':
    main()
