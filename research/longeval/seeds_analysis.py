"""Ten construction seeds for the held-out primary arms (design and prediction:
scripts/pace/LONGEVAL_HELDOUT_SEEDS.md, recorded before submission).

    python -m research.longeval.seeds_analysis INPUTS RELEASE FROZEN_EVIDENCE SEEDS_EVIDENCE OUTPUT.json

Combines the frozen three seeds (11, 29, 47) with the seven further seeds and reports, per primary
arm: D (the change in the ANN-minus-exact nDCG@10 gap from snapshot 1 to 3, GPT-OSS-120B labels, the
374-query panel) over all ten seeds with a two-level bootstrap (queries and seeds resampled), beside
the frozen three-seed value; the spread of D across seeds; how often calibration widened, and D for
seeds that widened and seeds that did not; and, for the HNSW arms, the calibration queries' recall
against the test queries' recall at the same width. Aggregates only.

As a check, the three frozen seeds must reproduce the frozen per-seed D of the sensitivity analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .heldout import ROLE, load_heldout
from .heldout_analysis import HNSW_MODES, IVF_MODES, PRIMARY_ARMS, collect, load_test_qrels, panel
from .heldout_seeds import EXTRA_SEEDS
from .heldout_sensitivity import FAMILY, ideal_dcg, ndcg10, ranked
from .release import checksum

FROZEN_SEEDS = (11, 29, 47)
ALL_SEEDS = FROZEN_SEEDS + tuple(EXTRA_SEEDS)
DRAWS = 10_000
RNG_SEED = 20261124
HNSW_DEFAULT_EF = 200
PREDICTION_MARGIN = 0.004
EPZERO_FROZEN_D = -0.0059


def d_matrix(runs, exact, qrels, population, arm, seeds):
    """D per query (rows) and seed (columns): [ANN − exact at snapshot 3] − [ANN − exact at snapshot 1]."""
    backend, mode, regime = arm
    columns = []
    for seed in seeds:
        gap = {}
        for step in (1, 3):
            values = []
            for q in population:
                grades = qrels[step].get(q, {})
                ideal = ideal_dcg(grades)
                values.append(ndcg10(ranked(runs[(backend, mode, seed, step, regime)][q]), grades, ideal)
                              - ndcg10(ranked(exact[step][q]), grades, ideal))
            gap[step] = np.array(values)
        columns.append(gap[3] - gap[1])
    return np.stack(columns, axis=1)


def two_level(values, *, draws=DRAWS, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n, k = values.shape
    sample = np.array([values[np.ix_(rng.integers(0, n, n), rng.integers(0, k, k))].mean() for _ in range(draws)])
    return np.quantile(sample, [.025, .975]).tolist()


def widths(reports, arm, seed):
    """The width calibration chose at each snapshot: ef for HNSW, nprobe for IVF."""
    backend, mode, _ = arm
    report = reports[(backend, mode, seed)]
    if backend == 'faiss_ivf':
        return [stage['selection']['nprobe'] for stage in report['snapshots']]
    return [stage['calibration']['effort'] for stage in report['snapshots']]


def widened(arm, chosen):
    """HNSW: calibration moved beyond ef 200 at any snapshot. IVF: nprobe at snapshot 3 above snapshot 1."""
    return any(w > HNSW_DEFAULT_EF for w in chosen) if arm[0] != 'faiss_ivf' else chosen[-1] > chosen[0]


def calibration_against_test(reports, arm, seeds):
    """HNSW arms: per snapshot, mean calibration-query recall at the chosen width (seeds with the
    observer only) and the test queries' recall of the matched run, averaged over seeds."""
    backend, mode, regime = arm
    out = {}
    for index, step in enumerate((1, 2, 3)):
        calibration, test, spread = [], [], []
        for seed in seeds:
            stage = reports[(backend, mode, seed)]['snapshots'][index]
            effort = stage['calibration']['effort']
            per_query = stage.get('diagnostic', {}).get('calibration_recall_per_query')
            if per_query is None:
                continue
            values = np.array(list(per_query[str(effort)].values()))
            calibration.append(values.mean())
            spread.append(values.std())
            test.append(np.mean(list(stage['runs'][regime]['ann_recall_per_query'].values())))
        if calibration:
            out[str(step)] = {'seeds': len(calibration), 'calibration_recall': float(np.mean(calibration)),
                              'test_recall': float(np.mean(test)), 'calibration_minus_test': float(np.mean(calibration) - np.mean(test)),
                              'calibration_query_sd': float(np.mean(spread))}
    return out


def analyze(inputs, release, frozen_evidence, seeds_evidence):
    queries, refs, _, loaded = load_heldout(inputs)
    ids = queries[ROLE].ids
    exact = {s: {q: refs[s - 1][q] for q in ids} for s in (1, 2, 3)}
    # The frozen seeds ran every mode; the further seeds only the primary ones (the design did not
    # repeat the identical-history rebuilds, identical to incremental maintenance in all 27 comparisons).
    expected = ({(b, m, s) for b in ('epzero', 'faiss_hnsw') for m in HNSW_MODES for s in FROZEN_SEEDS}
                | {('faiss_ivf', m, s) for m in IVF_MODES for s in FROZEN_SEEDS}
                | {(arm[0], arm[1], s) for arm in PRIMARY_ARMS.values() for s in EXTRA_SEEDS})
    paths = sorted(Path(frozen_evidence).rglob('report.json')) + sorted(Path(seeds_evidence).rglob('report.json'))
    reports, runs, excluded = collect(paths, ROLE, ids, exact, expected=expected)
    qrels_all, identity = load_test_qrels(release)
    qrels = qrels_all[FAMILY]
    population = panel(qrels, ids)
    out = {'kind': 'EXPLORATORY ten construction seeds for the held-out primary arms (aggregates only)',
           'design': 'scripts/pace/LONGEVAL_HELDOUT_SEEDS.md', 'source_sha256': checksum(__file__), 'inputs': loaded,
           'judgment_sha256': identity, 'family': FAMILY, 'panel': len(population), 'seeds': list(ALL_SEEDS),
           'excluded_reports': excluded, 'bootstrap': {'draws': DRAWS, 'seed': RNG_SEED, 'unit': 'queries and seeds'},
           'arms': {}}
    for name, arm in PRIMARY_ARMS.items():
        missing = [s for s in ALL_SEEDS if (arm[0], arm[1], s) not in reports]
        if missing:
            raise ValueError(f'{name}: no evidence for seeds {missing}')
        values = d_matrix(runs, exact, qrels, population, arm, ALL_SEEDS)
        per_seed = values.mean(axis=0)
        chosen = {s: widths(reports, arm, s) for s in ALL_SEEDS}
        wide = [i for i, s in enumerate(ALL_SEEDS) if widened(arm, chosen[s])]
        narrow = [i for i in range(len(ALL_SEEDS)) if i not in wide]
        block = {'D_ten_seeds': float(values.mean()), 'interval_95_queries_and_seeds': two_level(values),
                 'D_frozen_three_seeds': float(values[:, :3].mean()),
                 'D_by_seed': {str(s): round(float(v), 5) for s, v in zip(ALL_SEEDS, per_seed)},
                 'sd_across_seeds': float(per_seed.std(ddof=1)),
                 'calibrated_widths_by_seed': {str(s): chosen[s] for s in ALL_SEEDS},
                 'seeds_widened': [ALL_SEEDS[i] for i in wide],
                 'D_seeds_widened': float(values[:, wide].mean()) if wide else None,
                 'D_seeds_not_widened': float(values[:, narrow].mean()) if narrow else None}
        if arm[0] != 'faiss_ivf':
            block['calibration_against_test'] = calibration_against_test(reports, arm, EXTRA_SEEDS)
        out['arms'][name] = block
    epzero, faiss, ivf = (out['arms'][k] for k in ('epzero_hnsw', 'faiss_hnsw', 'ivf_frozen'))
    out['prediction'] = {
        'statement': 'FAISS and IVF stay within ±0.004 over ten seeds; Epzero\'s ten-seed D is smaller in magnitude than its three-seed -0.0059',
        'faiss_point_within_0.004': abs(faiss['D_ten_seeds']) <= PREDICTION_MARGIN,
        'faiss_interval_within_0.004': all(abs(x) <= PREDICTION_MARGIN for x in faiss['interval_95_queries_and_seeds']),
        'ivf_point_within_0.004': abs(ivf['D_ten_seeds']) <= PREDICTION_MARGIN,
        'ivf_interval_within_0.004': all(abs(x) <= PREDICTION_MARGIN for x in ivf['interval_95_queries_and_seeds']),
        'epzero_ten_seed_magnitude_below_three_seed': abs(epzero['D_ten_seeds']) < abs(EPZERO_FROZEN_D)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('inputs', 'release', 'frozen_evidence', 'seeds_evidence', 'output'):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    out = analyze(a.inputs, a.release, a.frozen_evidence, a.seeds_evidence)
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({name: {k: v for k, v in block.items() if k in ('D_ten_seeds', 'interval_95_queries_and_seeds',
                                                                       'D_frozen_three_seeds', 'sd_across_seeds', 'seeds_widened')}
                      for name, block in out['arms'].items()} | {'prediction': out['prediction']}, indent=2))


if __name__ == '__main__':
    main()
