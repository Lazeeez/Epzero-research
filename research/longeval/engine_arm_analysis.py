"""Pre-registered contrasts of the repaired-Epzero arm (design: scripts/pace/LONGEVAL_ENGINE_ARM.md).

    python -m research.longeval.engine_arm_analysis BUILDS OUTPUT.json [--confirm]

BUILDS holds `{arm}-{seed}/` folders from `engine_arm` (any number of job folders may be merged into
it). Per arm, seeds are averaged within each query; contrasts are paired by query, with a query
bootstrap. Margins are those of decision 3 in docs/arrival-bias-research.md:

  effect      the newest cohort's within-query excess (snapshot 3 against 1) is lower in `repaired`
              than in `grown`, the 95% interval of the difference excluding 0, and its point
              estimate in `repaired` is at most 1.0 percentage point;
  no harm     the upper 95% bound of the rise in the oldest cohort's miss rate is at most 0.75 points,
              and of the fall in recall@100 at most 0.5 points.

On the 100 confirmation queries the effect criterion decides; no harm is reported with its interval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .insertion_strata import risk_difference

ARMS = ('grown', 'repaired', 'rebuilt')
EFFECT_MAX_PP, OLDEST_MARGIN_PP, RECALL_MARGIN_PP = 1.0, 0.75, 0.5


def tables(builds, *, ef=200, prefix=''):
    """{arm: seed-averaged per-query table}, and the seeds found for each arm (all arms must share them)."""
    found = {}
    for folder in sorted(Path(builds).glob('*-*')):
        arm, seed = folder.name.rsplit('-', 1)
        if arm in ARMS and (folder / f'{prefix}per-query-ef{ef}.npy').exists():
            found.setdefault(arm, {})[int(seed)] = np.load(folder / f'{prefix}per-query-ef{ef}.npy').astype(float)
    if set(found) != set(ARMS) or len({tuple(sorted(s)) for s in found.values()}) != 1:
        raise ValueError(f'every arm needs the same seeds; found { {a: sorted(s) for a, s in found.items()} }')
    seeds = sorted(next(iter(found.values())))
    return {arm: np.mean([found[arm][s] for s in seeds], axis=0) for arm in ARMS}, seeds


def oldest(t):
    return t[:, 0, :, 1].sum() / t[:, 0, :, 0].sum()


def recall(t):
    return 1 - t[..., 1].sum() / t[..., 0].sum()


STATISTICS = {'newest_excess': lambda t: risk_difference(t, 3), 'oldest_miss': oldest, 'recall_at_100': recall}


def analyze(builds, *, ef=200, confirm=False, draws=10_000, seed=20261124):
    by_arm, seeds = tables(builds, ef=ef, prefix='confirm-' if confirm else '')
    nq = next(iter(by_arm.values())).shape[0]
    rows = np.random.default_rng(seed).integers(0, nq, (draws, nq))
    point = {arm: {name: 100 * float(f(t)) for name, f in STATISTICS.items()} for arm, t in by_arm.items()}
    contrasts = {}
    for a, b in (('repaired', 'grown'), ('rebuilt', 'grown'), ('repaired', 'rebuilt')):
        block = {}
        for name, f in STATISTICS.items():
            sample = np.array([100 * (f(by_arm[a][r]) - f(by_arm[b][r])) for r in rows])
            block[name] = {'value': point[a][name] - point[b][name],
                           'interval_95': np.quantile(sample, [.025, .975]).tolist()}
        contrasts[f'{a}_minus_{b}'] = block
    c = contrasts['repaired_minus_grown']
    decision = {
        'effect': bool(c['newest_excess']['interval_95'][1] < 0 and point['repaired']['newest_excess'] <= EFFECT_MAX_PP),
        'oldest_no_harm': bool(c['oldest_miss']['interval_95'][1] <= OLDEST_MARGIN_PP),
        'recall_no_harm': bool(-c['recall_at_100']['interval_95'][0] <= RECALL_MARGIN_PP)}
    decision['passed'] = decision['effect'] if confirm else all(decision.values())
    return {'kind': 'repaired-Epzero arm, pre-registered contrasts (aggregates only)', 'ef': ef,
            'queries': 'confirmation (100 training queries)' if confirm else 'test (381)', 'query_count': int(nq),
            'seeds': seeds, 'bootstrap': {'draws': draws, 'seed': seed, 'unit': 'query, seeds averaged within query'},
            'margins_pp': {'effect_max': EFFECT_MAX_PP, 'oldest': OLDEST_MARGIN_PP, 'recall': RECALL_MARGIN_PP},
            'arms': point, 'contrasts': contrasts, 'decision': decision}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('builds', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--ef', type=int, default=200)
    parser.add_argument('--confirm', action='store_true')
    a = parser.parse_args()
    out = analyze(a.builds, ef=a.ef, confirm=a.confirm)
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'arms': out['arms'], 'decision': out['decision']}, indent=2))


if __name__ == '__main__':
    main()
