"""Sensitivity of the held-out results to the choices a reader can question (EXPLORATORY).

Not part of FINAL_PROTOCOL_V1: the frozen decisions stand. Each analysis asks how far a result
depends on one choice, using only the verified held-out evidence:

1. bounds: what the primary intervals rule out, beside the frozen ±0.01 margin, and the relative
   change of the gap;
2. relevance_threshold: the gaps and their change when only grade 2 counts as relevant, and when any
   positive grade counts as fully relevant;
3. seed_bootstrap: intervals that resample construction seeds as well as queries;
4. label_noise: the gaps when each released grade is replaced by another draw of the same assessor,
   with the repeat behaviour measured in the judge pilot;
5. panel_exclusions: the test queries the fixed panel leaves out, and how much new content they hold;
6. freshness_by_difficulty: the fresh-content loss within strata of how hard a query already was for
   the index at snapshot 1.

Outputs hold aggregates only.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .heldout import ROLE, load_heldout
from .heldout_analysis import PRIMARY_ARMS, SECONDARY_ARMS, SEEDS, SNAPSHOTS, _estimate, _summary, collect, load_test_qrels, panel, score
from .release import checksum
from .storage import atomic_json

FAMILY = 'gpt-oss-120b'
DRAWS = 10_000
RNG_SEED = 20261124


# ------------------------------------------------------------------ nDCG@10 for repeated rescoring
def ranked(run_q):
    """Document IDs in the frozen evaluator's order: descending score, then descending ID."""
    return [d for d, _ in sorted(run_q, key=lambda x: (x[1], x[0]), reverse=True)]


def ndcg10(top, grades, ideal):
    """Linear-gain nDCG@10 of `top` (document IDs in rank order) under `grades`; `ideal` is the ideal DCG."""
    if ideal <= 0:
        return 0.0
    gain = sum(max(0, grades.get(d, 0)) / np.log2(rank + 2) for rank, d in enumerate(top[:10]))
    return gain / ideal


def ideal_dcg(grades):
    ordered = sorted((g for g in grades.values() if g > 0), reverse=True)[:10]
    return sum(g / np.log2(rank + 2) for rank, g in enumerate(ordered))


def gap_matrix(runs, exact, qrels, population, arm, *, grade_map=None):
    """D per query and seed for one arm: [ANN − exact at s3] − [ANN − exact at s1], with an optional grade map."""
    backend, mode, regime = arm
    cols = []
    for seed in SEEDS:
        per_step = {}
        for s in (1, 3):
            out = []
            for q in population:
                grades = qrels[s].get(q, {})
                if grade_map is not None:
                    grades = {d: grade_map(g) for d, g in grades.items()}
                ideal = ideal_dcg(grades)
                ann = ndcg10(ranked(runs[(backend, mode, seed, s, regime)][q]), grades, ideal)
                ref = ndcg10(ranked(exact[s][q]), grades, ideal)
                out.append(ann - ref)
            per_step[s] = np.array(out)
        cols.append(per_step[3] - per_step[1])
    return np.stack(cols, axis=1)


def gaps_by_snapshot(runs, exact, qrels, population, arm, *, grade_map=None):
    backend, mode, regime = arm
    out = []
    for s in SNAPSHOTS:
        values = []
        for q in population:
            grades = qrels[s].get(q, {})
            if grade_map is not None:
                grades = {d: grade_map(g) for d, g in grades.items()}
            ideal = ideal_dcg(grades)
            ref = ndcg10(ranked(exact[s][q]), grades, ideal)
            values.append(np.mean([ndcg10(ranked(runs[(backend, mode, seed, s, regime)][q]), grades, ideal) - ref
                                   for seed in SEEDS]))
        out.append(float(np.mean(values)))
    return out


def check_against_frozen(runs, exact, qrels, population):
    """The fast scorer must reproduce the frozen scorer (pytrec_eval-checked) exactly."""
    arm = PRIMARY_ARMS['epzero_hnsw']
    frozen = score(runs[(arm[0], arm[1], SEEDS[0], 3, arm[2])], qrels[3], population)
    fast = {}
    for q in population:
        grades = qrels[3].get(q, {})
        fast[q] = ndcg10(ranked(runs[(arm[0], arm[1], SEEDS[0], 3, arm[2])][q]), grades, ideal_dcg(grades))
    worst = max(abs(fast[q] - frozen[q]['nDCG@10']) for q in population)
    if worst > 1e-12:
        raise ValueError(f'fast nDCG@10 differs from the frozen scorer by {worst}')
    return worst


# ------------------------------------------------------------------ 1. bounds
def bounds(matrices, gaps):
    """For each primary arm: the frozen-style interval (95%), a 90% interval (two one-sided tests at 5%),
    the largest change each allows, and the relative change of the gap from snapshot 1 to 3."""
    out = {}
    for name, values in matrices.items():
        result = _summary(_estimate(list(range(len(values))), list(SEEDS), values))
        effects = values.mean(axis=1)
        rng = np.random.default_rng(RNG_SEED)
        boot = effects[rng.integers(0, len(effects), (DRAWS, len(effects)))].mean(axis=1)
        low90, high90 = np.quantile(boot, [.05, .95])
        low95, high95 = result['percentile_interval']
        out[name] = {'D': result['mean_difference'], 'interval_95': [low95, high95], 'interval_90': [float(low90), float(high90)],
                     'largest_change_allowed_95': float(max(abs(low95), abs(high95))),
                     'equivalent_within': {str(m): bool(-m < low90 and high90 < m) for m in (0.003, 0.004, 0.005, 0.0075, 0.01)},
                     'gap_by_snapshot': gaps[name],
                     'relative_change_of_gap': float(gaps[name][2] / gaps[name][0]) if gaps[name][0] else None}
    return out


# ------------------------------------------------------------------ 2. relevance threshold
def relevance_threshold(runs, exact, qrels, population):
    maps = {'official grades (0-2)': None,
            'strict: only grade 2 counts, as 1': lambda g: 1 if g >= 2 else 0,
            'binary: any positive grade counts as 1': lambda g: 1 if g >= 1 else 0}
    out = {}
    for label, grade_map in maps.items():
        block = {}
        for name, arm in PRIMARY_ARMS.items():
            values = gap_matrix(runs, exact, qrels, population, arm, grade_map=grade_map)
            result = _summary(_estimate(population, list(SEEDS), values))
            block[name] = {'gap_by_snapshot': gaps_by_snapshot(runs, exact, qrels, population, arm, grade_map=grade_map),
                           'D': result['mean_difference'], 'interval_95': result['percentile_interval']}
        out[label] = block
    return out


# ------------------------------------------------------------------ 3. seeds and queries
def seed_bootstrap(matrices, *, draws=DRAWS):
    """Resample the three seeds with replacement as well as the queries (two-level bootstrap)."""
    out = {}
    rng = np.random.default_rng(RNG_SEED)
    for name, values in matrices.items():
        n, k = values.shape
        sampled = np.empty(draws)
        for i in range(draws):
            seeds = rng.integers(0, k, k)
            rows = rng.integers(0, n, n)
            sampled[i] = values[np.ix_(rows, seeds)].mean()
        per_seed = values.mean(axis=0)
        out[name] = {'D': float(values.mean()), 'interval_95_queries_and_seeds': np.quantile(sampled, [.025, .975]).tolist(),
                     'D_by_seed': dict(zip(map(str, SEEDS), per_seed.round(5).tolist())),
                     'seed_spread': float(per_seed.max() - per_seed.min())}
    return out


# ------------------------------------------------------------------ 4. label noise
def repeat_model(pilot_report):
    """P(second grade | first grade) from the judge pilot's repeat of the organisers' own setting
    (no sampling parameter), on grades 0-2."""
    report = json.loads(Path(pilot_report).read_text())
    item = next(x for x in report['between_repeats'] if x['condition'] == 'provider_default')
    matrix = np.array(item['confusion_rows_first_columns_second'], dtype=float)[:3, :3]
    return matrix / matrix.sum(axis=1, keepdims=True), item['exact_agreement']


def label_noise(runs, exact, qrels, population, transition, *, replicates=200):
    """Each graded pair is regraded once per replicate from the repeat model; the same regrading is used at
    every snapshot (as in the release, a pair has one grade). Ungraded pairs stay ungraded."""
    rng = np.random.default_rng(RNG_SEED)
    pairs = sorted({(q, d) for s in SNAPSHOTS for q in population for d in qrels[s].get(q, {})})
    official = {}
    for s in SNAPSHOTS:
        for q in population:
            for d, g in qrels[s].get(q, {}).items():
                official[(q, d)] = max(0, min(2, g))
    results = defaultdict(list)
    cumulative = np.cumsum(transition, axis=1)
    for _ in range(replicates):
        u = rng.random(len(pairs))
        regraded = {p: int(np.searchsorted(cumulative[official[p]], u[i], side='right')) for i, p in enumerate(pairs)}
        noisy = {s: {q: {d: regraded[(q, d)] for d in qrels[s].get(q, {})} for q in population} for s in SNAPSHOTS}
        for name, arm in PRIMARY_ARMS.items():
            values = gap_matrix(runs, exact, noisy, population, arm)
            results[name].append(float(values.mean()))
    out = {}
    for name, draws in results.items():
        arr = np.array(draws)
        out[name] = {'D_mean_over_replicates': float(arr.mean()), 'D_sd': float(arr.std(ddof=1)),
                     'D_range_95': np.quantile(arr, [.025, .975]).tolist()}
    return {'replicates': replicates, 'pairs_regraded': len(pairs), 'arms': out}


# ------------------------------------------------------------------ 5 and 6. panel, freshness and difficulty
def panel_exclusions(ids, population, exact, first, qrels):
    excluded = [q for q in ids if q not in set(population)]
    share = {q: float(np.mean([first[d] > 1 for d, _ in exact[3][q][:10]])) for q in ids}
    positives = {q: [int(any(g > 0 for g in qrels[s].get(q, {}).values())) for s in SNAPSHOTS] for q in excluded}
    return {'test_queries': len(ids), 'panel': len(population), 'excluded': len(excluded),
            'excluded_positive_judgment_by_snapshot': [sum(v[i] for v in positives.values()) for i in range(3)],
            'new_content_share_in_exact_top10_at_s3': {
                'excluded_mean': float(np.mean([share[q] for q in excluded])) if excluded else None,
                'panel_mean': float(np.mean([share[q] for q in population])),
                'excluded_fresh_queries': sum(share[q] >= .5 for q in excluded),
                'panel_fresh_queries': sum(share[q] >= .5 for q in population)}}


def recall100(run_q, exact_q):
    truth = {d for d, _ in exact_q[:100]}
    return len(truth & {d for d, _ in run_q[:100]}) / len(truth)


def freshness_by_difficulty(runs, exact, qrels, population, first, *, draws=DRAWS):
    """Fresh-content queries (>=50% of the exact snapshot-3 top 10 inserted after snapshot 1) against the
    rest, within thirds of each query's neighbour recall at snapshot 1 at fixed ef 200 (before any new
    content existed): is the fresh-content loss more than these queries' difficulty?"""
    share = {q: float(np.mean([first[d] > 1 for d, _ in exact[3][q][:10]])) for q in population}
    out = {}
    rng = np.random.default_rng(RNG_SEED)
    for name, arm in PRIMARY_ARMS.items():
        backend, mode, regime = arm
        fixed = SECONDARY_ARMS[f'{name}_fixed_ef'] if f'{name}_fixed_ef' in SECONDARY_ARMS else SECONDARY_ARMS['ivf_frozen_fixed_nprobe']
        difficulty = {q: np.mean([recall100(runs[(fixed[0], fixed[1], seed, 1, fixed[2])][q], exact[1][q]) for seed in SEEDS])
                      for q in population}
        gap = {}
        for q in population:
            grades = qrels[3].get(q, {})
            ideal = ideal_dcg(grades)
            ref = ndcg10(ranked(exact[3][q]), grades, ideal)
            gap[q] = float(np.mean([ndcg10(ranked(runs[(backend, mode, seed, 3, regime)][q]), grades, ideal) - ref
                                    for seed in SEEDS]))
        ordered = sorted(population, key=lambda q: difficulty[q])
        thirds = np.array_split(np.array(ordered, dtype=object), 3)
        strata, contrasts = [], []
        for i, group in enumerate(thirds):
            fresh = [q for q in group if share[q] >= .5]
            old = [q for q in group if share[q] < .5]
            item = {'third_by_snapshot1_recall': ['hardest', 'middle', 'easiest'][i],
                    'snapshot1_recall_range': [float(difficulty[group[0]]), float(difficulty[group[-1]])],
                    'fresh_queries': len(fresh), 'old_queries': len(old),
                    'fresh_mean_gap': float(np.mean([gap[q] for q in fresh])) if fresh else None,
                    'old_mean_gap': float(np.mean([gap[q] for q in old])) if old else None}
            strata.append(item)
            if fresh and old:
                contrasts.append((np.array([gap[q] for q in fresh]), np.array([gap[q] for q in old])))
        # Stratum-weighted fresh-minus-old difference (weights: fresh queries per stratum), with a bootstrap within strata.
        weights = np.array([len(f) for f, _ in contrasts], dtype=float)
        weights /= weights.sum()
        point = float(sum(w * (f.mean() - o.mean()) for w, (f, o) in zip(weights, contrasts)))
        boot = np.empty(draws)
        for b in range(draws):
            boot[b] = sum(w * (f[rng.integers(0, len(f), len(f))].mean() - o[rng.integers(0, len(o), len(o))].mean())
                          for w, (f, o) in zip(weights, contrasts))
        raw_fresh = [gap[q] for q in population if share[q] >= .5]
        raw_old = [gap[q] for q in population if share[q] < .5]
        out[name] = {'strata': strata, 'unadjusted_fresh_minus_old': float(np.mean(raw_fresh) - np.mean(raw_old)),
                     'difficulty_adjusted_fresh_minus_old': point, 'interval_95': np.quantile(boot, [.025, .975]).tolist(),
                     'correlation_fresh_share_with_snapshot1_recall': float(np.corrcoef([share[q] for q in population],
                                                                                         [difficulty[q] for q in population])[0, 1])}
    return out


# ------------------------------------------------------------------ driver
def analyze(inputs, release, evidence, dense, pilot_report, output, *, noise_replicates=200):
    from .heldout_exploratory import cohorts
    inputs, release, evidence, output = map(Path, (inputs, release, evidence, output))
    queries, refs, _, loaded = load_heldout(inputs)
    ids = queries[ROLE].ids
    exact = {s: {q: refs[s - 1][q] for q in ids} for s in SNAPSHOTS}
    _, runs, _ = collect(sorted(evidence.rglob('report.json')), ROLE, ids, exact)
    qrels_all, identity = load_test_qrels(release)
    qrels = qrels_all[FAMILY]
    population = panel(qrels, ids)
    worst = check_against_frozen(runs, exact, qrels, population)
    first = cohorts(dense)
    matrices = {name: gap_matrix(runs, exact, qrels, population, arm) for name, arm in PRIMARY_ARMS.items()}
    gaps = {name: gaps_by_snapshot(runs, exact, qrels, population, arm) for name, arm in PRIMARY_ARMS.items()}
    transition, agreement = repeat_model(pilot_report)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'completed', 'kind': 'EXPLORATORY sensitivity of the held-out results (not part of FINAL_PROTOCOL_V1)',
              'source_sha256': checksum(__file__), 'inputs': loaded, 'judgment_sha256': identity, 'family': FAMILY,
              'panel': len(population), 'fast_scorer_max_difference_from_frozen': worst,
              'bounds': bounds(matrices, gaps),
              'relevance_threshold': relevance_threshold(runs, exact, qrels, population),
              'seed_bootstrap': seed_bootstrap(matrices),
              'label_noise': {'repeat_model': transition.round(4).tolist(), 'pilot_repeat_exact_agreement': agreement,
                              'pilot_report_sha256': checksum(pilot_report),
                              **label_noise(runs, exact, qrels, population, transition, replicates=noise_replicates)},
              'panel_exclusions': panel_exclusions(ids, population, exact, first, qrels),
              'freshness_by_difficulty': freshness_by_difficulty(runs, exact, qrels, population, first)}
    atomic_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('inputs', 'release', 'evidence', 'dense', 'pilot_report', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--noise-replicates', type=int, default=200)
    a = parser.parse_args()
    r = analyze(a.inputs, a.release, a.evidence, a.dense, a.pilot_report, a.output, noise_replicates=a.noise_replicates)
    print(json.dumps({k: r[k] for k in ('panel', 'fast_scorer_max_difference_from_frozen')}, indent=2))


if __name__ == '__main__':
    main()
