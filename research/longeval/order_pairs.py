"""Paired, target-level analysis of insertion order (exploratory follow-up; design in
scripts/pace/LONGEVAL_INSERTION_ORDER_IDS.md).

The builds of one implementation share the documents, the queries and the exact top-100 targets;
only the insertion order differs. Each (query, exact target) pair is therefore observed in every
build. For two orders built with the same seed this module counts, by arrival cohort, the pairs
found in both, missed in both, rescued (missed in the first order, found in the second) and newly
lost, and estimates the paired change in miss risk with a query bootstrap (seeds averaged within
query). Ratios near one can hide cancellation; these counts cannot.

    python -m research.longeval.order_pairs SAMPLE BUILDS OUTPUT [--ef 200] [--confirm]

SAMPLE holds `cohort.npy` and the exact targets; BUILDS holds `{order}-{seed}/labels-ef{ef}.npy`
(`confirm-labels-ef{ef}.npy` with --confirm). Output is aggregates only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ORDERS = ('appended', 'shuffled', 'reversed')


def missed(labels, targets):
    """[query, rank] booleans: is exact target `targets[q, r]` absent from `labels[q]`?"""
    if labels.shape[0] != targets.shape[0]:
        raise ValueError('labels and targets cover different queries')
    out = np.empty(targets.shape, dtype=bool)
    for q in range(targets.shape[0]):
        out[q] = ~np.isin(targets[q], labels[q])
    return out


def transitions(first, second, cohort_of_target):
    """Per query and cohort: counts of (found both, missed both, rescued, newly lost) from `first` to `second`."""
    out = np.zeros((first.shape[0], 3, 4), dtype=np.int64)
    kinds = (~first & ~second, first & second, first & ~second, ~first & second)
    for c in (1, 2, 3):
        mask = cohort_of_target == c
        for k, kind in enumerate(kinds):
            out[:, c - 1, k] = (kind & mask).sum(axis=1)
    return out


def _risk_change(table, cohort):
    """Change in miss risk of one cohort from the first order to the second: (lost - rescued) / eligible pairs."""
    eligible = table[:, cohort - 1].sum()
    return float((table[:, cohort - 1, 3].sum() - table[:, cohort - 1, 2].sum()) / eligible) if eligible else float('nan')


def compare(sample, builds, first, second, *, ef=200, confirm=False, draws=5000, seed=20261123):
    sample, builds = Path(sample), Path(builds)
    prefix = 'confirm-' if confirm else ''
    targets = np.load(sample / f'{prefix}exact-top100.npy')
    cohort_of_target = np.load(sample / 'cohort.npy')[targets]
    seeds = sorted({int(p.name.rsplit('-', 1)[1]) for p in builds.glob(f'{first}-*')}
                   & {int(p.name.rsplit('-', 1)[1]) for p in builds.glob(f'{second}-*')})
    if not seeds:
        raise ValueError(f'no seed built in both orders {first} and {second}')
    per_seed = {}
    for s in seeds:
        a = missed(np.load(builds / f'{first}-{s}' / f'{prefix}labels-ef{ef}.npy'), targets)
        b = missed(np.load(builds / f'{second}-{s}' / f'{prefix}labels-ef{ef}.npy'), targets)
        per_seed[s] = transitions(a, b, cohort_of_target)
    mean = np.mean([per_seed[s] for s in seeds], axis=0)            # seeds averaged within query
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, mean.shape[0], (draws, mean.shape[0]))
    result = {'first': first, 'second': second, 'ef': ef, 'queries': int(mean.shape[0]), 'seeds': seeds,
              'confirmation_queries': confirm, 'cohorts': {}}
    for c in (1, 2, 3):
        boot = np.array([_risk_change(mean[r], c) for r in rows])
        counts = mean[:, c - 1].sum(axis=0)
        result['cohorts'][str(c)] = {
            'eligible_pairs_per_build': float(counts.sum()),
            'found_both': float(counts[0]), 'missed_both': float(counts[1]),
            'rescued': float(counts[2]), 'newly_lost': float(counts[3]),
            'risk_change_pp': 100 * _risk_change(mean, c),
            'interval_95_pp': (100 * np.nanquantile(boot, [.025, .975])).tolist(),
            'per_seed_risk_change_pp': {str(s): 100 * _risk_change(per_seed[s], c) for s in seeds}}
    # The newest cohort's change minus the oldest's: how much the order moves the gap between them.
    gap = np.array([_risk_change(mean[r], 3) - _risk_change(mean[r], 1) for r in rows])
    result['gap_change_pp'] = {'value': 100 * (_risk_change(mean, 3) - _risk_change(mean, 1)),
                               'interval_95': (100 * np.nanquantile(gap, [.025, .975])).tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('sample', type=Path)
    parser.add_argument('builds', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--ef', type=int, default=200)
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    out = {'kind': 'EXPLORATORY paired target-level insertion-order analysis (aggregates only)',
           'contrasts': [compare(args.sample, args.builds, a, b, ef=args.ef, confirm=args.confirm)
                         for a, b in (('appended', 'reversed'), ('appended', 'shuffled'), ('shuffled', 'reversed'))]}
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
