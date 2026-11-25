"""Analysis C of the order rerun: what arrival order costs in relevance, with the judging pool graded in full.

    python -m research.longeval.order_relevance ORDER_BUILDS ORDER_METADATA RELEASE GRADING_INPUTS GRADING_OUTPUT OUTPUT

Designed in scripts/pace/LONGEVAL_INSERTION_ORDER_IDS.md (C) before the order rerun ran, to be computed once
the E1 judging pool, extended with the rerun's top-10 substitutions, was graded (job 7298863, amendment 5).
Test queries of the fixed panel, snapshot 3, ef 200, top 10, per implementation and insertion order (seeds
averaged within query):

- the share of the exact top 10's relevant documents that the build drops, by arrival group;
- the documents the build puts in their place: how many, how many officially graded, their local grades;
- net ranked gain, nDCG@10 of the build minus nDCG@10 of exact search, under the official grades and under the
  local full grades, and its change as arrived minus reversed (paired query bootstrap, 10,000 draws).

ORDER_BUILDS holds `<backend>/<order>-<seed>/labels-ef200.npy`; ORDER_METADATA one job's `doc-ids.txt.gz`,
`query-ids.json`, `exact-top100.npy` and `cohort.npy`; GRADING_INPUTS and GRADING_OUTPUT the `primary` and
`substitutions` folders of the grading package and of its job. The output is aggregates only.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from . import coverage
from .evaluation import evaluate
from .heldout_analysis import SEEDS, _estimate, _summary, load_test_qrels, panel

ORDERS = ('appended', 'shuffled', 'reversed')
BACKENDS = ('epzero', 'faiss_hnsw')
WIDTH, TOP, STEP = 200, 10, 3
FAMILY = 'gpt-oss-120b'


def _load_metadata(metadata):
    metadata = Path(metadata)
    with gzip.open(metadata / 'doc-ids.txt.gz', 'rt', encoding='utf-8') as handle:
        docs = handle.read().split('\n')[:-1]
    queries = json.loads((metadata / 'query-ids.json').read_text())
    exact = np.load(metadata / 'exact-top100.npy')[:, :TOP]
    cohort = np.load(metadata / 'cohort.npy')
    return docs, queries, exact, cohort


def _grades(grading_inputs, grading_output):
    """Local grades and the scoring universe at snapshot 3, from both grading folders."""
    grades, universe = {}, {}
    for folder in ('primary', 'substitutions'):
        found, _ = coverage.local_grades(Path(grading_inputs) / folder, Path(grading_output) / folder)
        grades.update(found)
        stored = json.loads((Path(grading_inputs) / folder / 'universe.json').read_text())
        for query, docs in stored['snapshots'].get(str(STEP), {}).items():
            universe.setdefault(query, set()).update(docs)
    return grades, universe


def _run(positions, docs):
    """A ranked list as the evaluator reads it: the document at rank r scores TOP - r."""
    return [(docs[p], float(TOP - r)) for r, p in enumerate(positions) if p >= 0]


def analyze(order_builds, metadata, release, grading_inputs, grading_output, draws=10_000):
    docs, queries, exact, cohort = _load_metadata(metadata)
    row_of = {q: i for i, q in enumerate(queries)}
    qrels, _ = load_test_qrels(Path(release) / 'files')
    official_all = qrels[FAMILY]
    population = panel(official_all, queries)
    grades, universe = _grades(grading_inputs, grading_output)
    full = {q: {d: grades[(q, d)] for d in universe.get(q, ()) if (q, d) in grades} for q in population}
    official = {q: dict(official_all[STEP].get(q, {})) for q in population}
    tables = {'official': official, 'local_full': full}

    exact_runs = {q: _run(exact[row_of[q]], docs) for q in population}
    exact_ndcg = {name: evaluate(exact_runs, table, population, population='explicit')['per_query'] for name, table in tables.items()}
    out = {'kind': 'Analysis C of the order rerun (LONGEVAL_INSERTION_ORDER_IDS.md): relevance consequence, aggregates only',
           'snapshot': STEP, 'width': WIDTH, 'top': TOP, 'queries': len(population), 'seeds': list(SEEDS),
           'bootstrap': {'draws': draws, 'unit': 'query, seeds averaged within query'}, 'backends': {}}
    for backend in BACKENDS:
        block = {'orders': {}, 'as_arrived_minus_reversed': {}}
        gains = {}
        for order in ORDERS:
            per_seed = {name: [] for name in tables}
            dropped = {name: np.zeros((3, 2)) for name in tables}  # per cohort: relevant in exact top 10, dropped
            substitutes = {'count': 0, 'officially_graded': 0, 'local_relevant': 0, 'local_grade_sum': 0.0}
            for seed in SEEDS:
                labels = np.load(Path(order_builds) / backend / f'{order}-{seed}' / f'labels-ef{WIDTH}.npy')[:, :TOP]
                runs = {q: _run(labels[row_of[q]], docs) for q in population}
                for name, table in tables.items():
                    ndcg = evaluate(runs, table, population, population='explicit')['per_query']
                    per_seed[name].append([ndcg[q]['nDCG@10'] - exact_ndcg[name][q]['nDCG@10'] for q in population])
                for q in population:
                    row = row_of[q]
                    returned = set(int(p) for p in labels[row] if p >= 0)
                    top = [int(p) for p in exact[row]]
                    for name, table in tables.items():
                        for p in top:
                            if table[q].get(docs[p], 0) > 0:
                                c = int(cohort[p]) - 1
                                dropped[name][c, 0] += 1
                                dropped[name][c, 1] += p not in returned
                    for p in returned - set(top):
                        d = docs[p]
                        substitutes['count'] += 1
                        substitutes['officially_graded'] += d in official[q]
                        g = full[q].get(d)
                        if g is not None:
                            substitutes['local_relevant'] += g > 0
                            substitutes['local_grade_sum'] += g
            gains[order] = {name: np.array(values).T for name, values in per_seed.items()}  # [query, seed]
            n = len(SEEDS)
            block['orders'][order] = {
                'net_gain_ndcg10': {name: float(m.mean()) for name, m in gains[order].items()},
                'relevant_dropped_share_by_arrival_group': {
                    name: {str(c + 1): (float(v[c, 1] / v[c, 0]) if v[c, 0] else None) for c in range(3)}
                    for name, v in dropped.items()},
                'relevant_in_exact_top10_by_arrival_group': {name: [int(v[c, 0] / n) for c in range(3)] for name, v in dropped.items()},
                'substitutes_per_query': substitutes['count'] / n / len(population),
                'substitutes_officially_graded_share': substitutes['officially_graded'] / max(substitutes['count'], 1),
                'substitutes_local_relevant_share': substitutes['local_relevant'] / max(substitutes['count'], 1),
                'substitutes_mean_local_grade': substitutes['local_grade_sum'] / max(substitutes['count'], 1)}
        for name in tables:
            change = _summary(_estimate(population, list(SEEDS), gains['appended'][name] - gains['reversed'][name]))
            block['as_arrived_minus_reversed'][name] = {k: change[k] for k in ('mean_difference', 'percentile_interval',
                                                                                'two_sided_randomization_p')}
        out['backends'][backend] = block
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('order_builds', 'metadata', 'release', 'grading_inputs', 'grading_output', 'output'):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    result = analyze(a.order_builds, a.metadata, a.release, a.grading_inputs, a.grading_output)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2)[:6000])


if __name__ == '__main__':
    main()
