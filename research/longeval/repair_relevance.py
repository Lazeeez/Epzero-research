"""Analysis R2: what the arrival repair does to ranked relevance (design: docs/analysis-r2-repair-relevance.md,
recorded before this was computed).

    python -m research.longeval.repair_relevance ENGINE_BUILDS ORDER_METADATA RELEASE GRADING_INPUTS GRADING_OUTPUT OUTPUT

ENGINE_BUILDS holds `<arm>-<seed>/labels-ef200.npy` of the full-scale engine arm (run 3); the other inputs are
Analysis C's (`order_relevance.py`). Aggregates only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import evaluate
from .heldout_analysis import SEEDS, _estimate, _summary, load_test_qrels, panel
from .order_relevance import FAMILY, STEP, TOP, WIDTH, _grades, _load_metadata, _run

ARMS = ('grown', 'repaired8w', 'rebuilt')
CONTRASTS = (('repaired8w', 'grown'), ('rebuilt', 'grown'))
BOUND_GRADE = 3


def analyze(engine_builds, metadata, release, grading_inputs, grading_output, draws=10_000):
    docs, queries, exact, cohort = _load_metadata(metadata)
    row_of = {q: i for i, q in enumerate(queries)}
    qrels, _ = load_test_qrels(Path(release) / 'files')
    population = panel(qrels[FAMILY], queries)
    grades, universe = _grades(grading_inputs, grading_output)
    official = {q: dict(qrels[FAMILY][STEP].get(q, {})) for q in population}
    full = {q: {d: grades[(q, d)] for d in universe.get(q, ()) if (q, d) in grades} for q in population}
    labels = {(arm, s): np.load(Path(engine_builds) / f'{arm}-{s}' / f'labels-ef{WIDTH}.npy')[:, :TOP]
              for arm in ARMS for s in SEEDS}
    ungraded = sorted({(q, docs[p]) for (arm, s), lab in labels.items() for q in population for p in lab[row_of[q]]
                       if p >= 0 and (q, docs[p]) not in grades and docs[p] not in official[q]})
    bound = {q: dict(full[q]) for q in population}
    for q, d in ungraded:
        bound[q][d] = BOUND_GRADE
    tables = {'official': official, 'local_full': full, 'local_full_ungraded_at_3': bound}
    exact_runs = {q: _run(exact[row_of[q]], docs) for q in population}
    exact_ndcg = {n: evaluate(exact_runs, t, population, population='explicit')['per_query'] for n, t in tables.items()}
    gains, out = {}, {'kind': 'Analysis R2 (docs/analysis-r2-repair-relevance.md): the repair and ranked relevance, aggregates only',
                      'snapshot': STEP, 'width': WIDTH, 'top': TOP, 'queries': len(population), 'seeds': list(SEEDS),
                      'ungraded_pairs': len(ungraded), 'bootstrap': {'draws': draws, 'unit': 'query, seeds averaged within query'},
                      'arms': {}, 'contrasts': {}}
    for arm in ARMS:
        per_seed = {n: [] for n in tables}
        dropped = {n: np.zeros((3, 2)) for n in tables}
        subs = {'count': 0, 'officially_graded': 0, 'local_relevant': 0}
        for s in SEEDS:
            lab = labels[(arm, s)]
            runs = {q: _run(lab[row_of[q]], docs) for q in population}
            for n, t in tables.items():
                ndcg = evaluate(runs, t, population, population='explicit')['per_query']
                per_seed[n].append([ndcg[q]['nDCG@10'] - exact_ndcg[n][q]['nDCG@10'] for q in population])
            for q in population:
                returned = {int(p) for p in lab[row_of[q]] if p >= 0}
                top = [int(p) for p in exact[row_of[q]]]
                for n, t in tables.items():
                    for p in top:
                        if t[q].get(docs[p], 0) > 0:
                            dropped[n][int(cohort[p]) - 1, 0] += 1
                            dropped[n][int(cohort[p]) - 1, 1] += p not in returned
                for p in returned - set(top):
                    subs['count'] += 1
                    subs['officially_graded'] += docs[p] in official[q]
                    subs['local_relevant'] += full[q].get(docs[p], 0) > 0
        gains[arm] = {n: np.array(v).T for n, v in per_seed.items()}
        out['arms'][arm] = {
            'net_gain_ndcg10': {n: float(m.mean()) for n, m in gains[arm].items()},
            'relevant_dropped_share_by_arrival_group': {n: {str(c + 1): float(v[c, 1] / v[c, 0]) for c in range(3)} for n, v in dropped.items()},
            'substitutes_per_query': subs['count'] / len(SEEDS) / len(population),
            'substitutes_officially_graded_share': subs['officially_graded'] / max(subs['count'], 1),
            'substitutes_local_relevant_share': subs['local_relevant'] / max(subs['count'], 1)}
    for a, b in CONTRASTS:
        block = {}
        for n in tables:
            change = _summary(_estimate(population, list(SEEDS), gains[a][n] - gains[b][n]))
            block[n] = {k: change[k] for k in ('mean_difference', 'percentile_interval', 'two_sided_randomization_p')}
        out['contrasts'][f'{a}_minus_{b}'] = block
    c = out['contrasts']['repaired8w_minus_grown']
    out['prediction'] = {
        'interval_within_0.01': {n: all(abs(x) <= 0.01 for x in c[n]['percentile_interval']) for n in ('official', 'local_full')},
        'repaired_drops_fewer_newest': {n: out['arms']['repaired8w']['relevant_dropped_share_by_arrival_group'][n]['3']
                                        < out['arms']['grown']['relevant_dropped_share_by_arrival_group'][n]['3'] for n in ('official', 'local_full')},
        'bound_moves_below_0.002': max(abs(out['contrasts'][k]['local_full_ungraded_at_3']['mean_difference']
                                           - out['contrasts'][k]['local_full']['mean_difference']) for k in out['contrasts']) < 0.002}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('engine_builds', 'metadata', 'release', 'grading_inputs', 'grading_output', 'output'):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    out = analyze(a.engine_builds, a.metadata, a.release, a.grading_inputs, a.grading_output)
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k in ('ungraded_pairs', 'contrasts', 'prediction')}, indent=2))


if __name__ == '__main__':
    main()
