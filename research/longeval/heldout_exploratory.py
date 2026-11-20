"""Post-freeze EXPLORATORY analyses of the held-out evidence (not part of FINAL_PROTOCOL_V1).

1. Insertion cohort: are exact neighbours that were inserted later missed more often?
   Miss rates are compared within exact-rank bands, because newer documents can sit at
   different depths of the top 100, and deeper neighbours are missed more often.
2. Replacement relevance: when approximate search drops an exact top-10 neighbour, how
   relevant (GPT-OSS-120B) is the document that takes its place?

Outputs carry no inference claim; they describe the frozen evidence.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dense import ARTIFACTS
from .evaluation import ann_recall, read_run
from .heldout import ROLE, load_heldout
from .heldout_analysis import PRIMARY_FAMILY, SNAPSHOTS, collect, load_test_qrels, panel
from .mapped import MappedSnapshot
from .release import checksum
from .storage import atomic_json

BANDS = ((1, 10), (11, 50), (51, 100))
SELECTED = (('epzero', 'incremental', 'frozen'), ('epzero', 'incremental', 'matched_recall'),
            ('faiss_hnsw', 'incremental', 'frozen'), ('faiss_hnsw', 'incremental', 'matched_recall'),
            ('faiss_ivf', 'frozen_incremental', 'current_calibration'),
            ('faiss_ivf', 'frozen_incremental', 'frozen_snapshot1'),
            ('faiss_ivf', 'refreshed_rebuild', 'current_calibration'))


def cohorts(dense):
    """Snapshot at which each document first appears (1, 2 or 3)."""
    first = {}
    for artifact in ARTIFACTS:
        step = artifact['snapshot']
        with MappedSnapshot(Path(dense) / f'snapshot-{step}', expected_archive_sha256=artifact['archive_sha256']) as m:
            for doc in m.ids:
                first.setdefault(doc, step)
    return first


def cohort_misses(runs, exact, first):
    """Counts of exact neighbours and misses by (arm, snapshot, cohort, rank band), pooled over seeds and queries."""
    table = defaultdict(lambda: [0, 0])
    for (backend, mode, seed, step, regime), run in runs.items():
        arm = (backend, mode, regime)
        if arm not in SELECTED:
            continue
        for q, reference in exact[step].items():
            returned = {d for d, _ in run[q]}
            for rank, (doc, _) in enumerate(reference, 1):
                band = next(b for b in BANDS if b[0] <= rank <= b[1])
                cell = table[(arm, step, first[doc], band)]
                cell[0] += 1
                cell[1] += doc not in returned
    rows = []
    for (arm, step, cohort, band), (count, missed) in sorted(table.items()):
        rows.append({'arm': '/'.join(arm), 'snapshot': step, 'inserted_at': cohort, 'exact_rank_band': f'{band[0]}-{band[1]}',
                     'exact_neighbours': count, 'missed': missed, 'miss_rate': missed / count})
    return rows


def replacements(runs, exact, qrels, population):
    """Mean GPT-OSS-120B grade of exact top-10 neighbours the ANN dropped vs documents it put there instead."""
    out = []
    for arm in SELECTED:
        for step in SNAPSHOTS:
            dropped, added, kept = [], [], []
            judged_added = 0
            for (backend, mode, seed, s, regime), run in runs.items():
                if (backend, mode, regime) != arm or s != step:
                    continue
                for q in population:
                    grades = qrels[step].get(q, {})
                    e10 = [d for d, _ in exact[step][q][:10]]
                    a10 = [d for d, _ in run[q][:10]]
                    for d in set(e10) - set(a10):
                        dropped.append(max(0, grades.get(d, 0)))
                    for d in set(a10) - set(e10):
                        added.append(max(0, grades.get(d, 0)))
                        judged_added += d in grades
                    kept.extend(max(0, grades.get(d, 0)) for d in set(e10) & set(a10))
            out.append({'arm': '/'.join(arm), 'snapshot': step, 'dropped_count': len(dropped), 'added_count': len(added),
                        'dropped_mean_grade': float(np.mean(dropped)) if dropped else None,
                        'added_mean_grade': float(np.mean(added)) if added else None,
                        'added_judged_fraction': judged_added / len(added) if added else None,
                        'kept_mean_grade': float(np.mean(kept)),
                        'dropped_positive_fraction': float(np.mean([g > 0 for g in dropped])) if dropped else None,
                        'added_positive_fraction': float(np.mean([g > 0 for g in added])) if added else None})
    return out


def explore(inputs, release, evidence, dense, output):
    inputs, release, evidence, output = map(Path, (inputs, release, evidence, output))
    queries, refs, _, loaded = load_heldout(inputs)
    ids = queries[ROLE].ids
    exact = {step: {q: refs[step - 1][q] for q in ids} for step in SNAPSHOTS}
    _, runs, excluded = collect(sorted(evidence.rglob('report.json')), ROLE, ids, exact)
    qrels, identity = load_test_qrels(release)
    population = panel(qrels[PRIMARY_FAMILY], ids)
    first = cohorts(dense)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'completed', 'kind': 'EXPLORATORY post-freeze analysis; not part of FINAL_PROTOCOL_V1',
              'source_sha256': checksum(__file__), 'inputs': loaded, 'judgment_sha256': identity,
              'panel_count': len(population), 'excluded_attempts': excluded,
              'cohort_sizes': {str(s): sum(v == s for v in first.values()) for s in SNAPSHOTS},
              'cohort_misses': cohort_misses(runs, exact, first),
              'replacements': replacements(runs, exact, qrels[PRIMARY_FAMILY], population)}
    atomic_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('inputs', 'release', 'evidence', 'dense', 'output'):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    r = explore(a.inputs, a.release, a.evidence, a.dense, a.output)
    print(json.dumps({k: r[k] for k in ('status', 'panel_count', 'cohort_sizes')}, indent=2))


if __name__ == '__main__':
    main()


def development_order(root, dense, output):
    """Insertion order vs document property: cohort miss rates for incremental vs shuffled-order
    rebuilds on the 50 development queries (shuffled runs exist only in development)."""
    from .query_diagnostic import validate_inputs
    root, output = Path(root), Path(output)
    build = root / 'build'
    contract, _ = validate_inputs(build / 'longeval-training-query-inputs-01')
    ids = contract['roles']['development']
    exact = {}
    for step in SNAPSHOTS:
        folder = build / 'longeval-query-sensitivity-01' / f'snapshot-{step}'
        receipt = json.loads((folder / 'report.json').read_text())
        if receipt['artifacts']['float32.run'] != checksum(folder / 'float32.run'):
            raise ValueError('development exact reference checksum mismatch')
        run = read_run(folder / 'float32.run')
        exact[step] = {q: run[q] for q in ids}
    expected = {(b, m, s) for b in ('epzero', 'faiss_hnsw') for m in ('incremental', 'shuffled') for s in (11, 29, 47)}
    paths = sorted(build.glob('longeval-pace-trajectory-evidence-*/*/trajectory/report.json'))
    _, runs, _ = collect(paths, 'development', ids, exact, expected=expected)
    first = cohorts(dense)
    global SELECTED
    selected, SELECTED = SELECTED, tuple((b, m, 'frozen') for b in ('epzero', 'faiss_hnsw') for m in ('incremental', 'shuffled'))
    try:
        rows = cohort_misses(runs, exact, first)
    finally:
        SELECTED = selected
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'completed', 'kind': 'EXPLORATORY development analysis; 50 development queries, ef 200',
              'source_sha256': checksum(__file__), 'cohort_misses': rows}
    atomic_json(output / 'report.json', report)
    return report


def cohort_ratio(runs, exact, first, arm, step, *, draws=10_000, seed=20261120):
    """Rank-standardized miss ratio of later cohorts (inserted after snapshot 1) vs snapshot-1 documents,
    with a query bootstrap (seeds pooled within query). Expected misses use snapshot-1 band rates."""
    per_query = {}
    for (backend, mode, s, st, regime), run in runs.items():
        if (backend, mode, regime) != arm or st != step:
            continue
        for q, reference in exact[step].items():
            returned = {d for d, _ in run[q]}
            cells = per_query.setdefault(q, defaultdict(lambda: [0, 0]))
            for rank, (doc, _) in enumerate(reference, 1):
                band = next(b for b in BANDS if b[0] <= rank <= b[1])
                cell = cells[(first[doc] > 1, band)]
                cell[0] += 1
                cell[1] += doc not in returned
    queries = sorted(per_query)

    def ratio(sample):
        total = defaultdict(lambda: [0, 0])
        for q in sample:
            for key, (n, m) in per_query[q].items():
                total[key][0] += n
                total[key][1] += m
        observed = sum(total[(True, b)][1] for b in BANDS)
        expected = sum(total[(True, b)][0] * total[(False, b)][1] / max(1, total[(False, b)][0]) for b in BANDS)
        return observed / expected

    rng = np.random.default_rng(seed)
    boot = [ratio([queries[i] for i in rng.integers(0, len(queries), len(queries))]) for _ in range(draws)]
    return {'arm': '/'.join(arm), 'snapshot': step, 'ratio': ratio(queries),
            'interval_95': np.quantile(boot, [.025, .975]).tolist(), 'queries': len(queries), 'draws': draws}


def condensed_gaps(runs, exact, qrels_by_snapshot, population):
    """Condensed-list nDCG@10: unjudged documents removed from every ranking before scoring.
    Returns per primary arm the gap estimate at each snapshot and D (s3 - s1), primary inference settings."""
    from .evaluation import evaluate
    from .heldout_analysis import PRIMARY_ARMS, SEEDS, _estimate, _summary

    def ndcg(run, step):
        judged = qrels_by_snapshot[step]
        condensed = {q: [(d, s) for d, s in run[q] if d in judged.get(q, {})] for q in population}
        return {q: v['nDCG@10'] for q, v in evaluate(condensed, judged, population, population='explicit')['per_query'].items()}

    base = {s: ndcg(exact[s], s) for s in SNAPSHOTS}
    out = {}
    for name, (backend, mode, regime) in PRIMARY_ARMS.items():
        gaps = {}
        for s in SNAPSHOTS:
            scores = {seed: ndcg(runs[(backend, mode, seed, s, regime)], s) for seed in SEEDS}
            gaps[s] = np.array([[scores[seed][q] - base[s][q] for seed in SEEDS] for q in population])
        out[name] = {'gap_by_snapshot': {s: _summary(_estimate(population, list(SEEDS), gaps[s])) for s in SNAPSHOTS},
                     'gap_change_D': _summary(_estimate(population, list(SEEDS), gaps[3] - gaps[1]))}
    return out


def freshness_gaps(per_query_csv, exact, first, population, *, step=3, family='gpt-oss-120b', draws=10_000, seed=20261120):
    """Snapshot-`step` ANN-minus-exact nDCG@10 for queries whose exact top 10 is mostly new content
    (inserted after snapshot 1) versus mostly old, per primary arm; seeds averaged within query."""
    import csv
    from .heldout_analysis import PRIMARY_ARMS
    share = {q: np.mean([first[d] > 1 for d, _ in exact[step][q][:10]]) for q in population}
    groups = {'fresh (>=50% new in exact top 10)': [q for q in population if share[q] >= .5],
              'old (<50% new)': [q for q in population if share[q] < .5]}
    gaps = defaultdict(dict)
    with open(per_query_csv, newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['family'] != family or int(row['snapshot']) != step or row['backend'] == 'exact':
                continue
            gaps[(row['backend'], row['mode'], row['regime'])].setdefault(row['query_id'], []).append(
                float(row['ANN_minus_exact_nDCG@10']))
    rng = np.random.default_rng(seed)
    out = {'query_counts': {k: len(v) for k, v in groups.items()}, 'arms': {}}
    for name, arm in PRIMARY_ARMS.items():
        per = {q: float(np.mean(v)) for q, v in gaps[arm].items()}
        item = {}
        for label, qs in groups.items():
            values = np.array([per[q] for q in qs])
            boot = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
            item[label] = {'mean_gap': float(values.mean()), 'interval_95': np.quantile(boot, [.025, .975]).tolist()}
        out['arms'][name] = item
    return out


def extension(inputs, release, primary_evidence, extension_evidence, dense, output, *, draws=2000):
    """Analyse the exploratory held-out extension: shuffled-order arm and ef sweep (GPT-OSS-120B panel)."""
    from .heldout_analysis import SEEDS, _estimate, _summary, score
    inputs, release, extension_evidence, output = map(Path, (inputs, release, extension_evidence, output))
    queries, refs, _, loaded = load_heldout(inputs)
    ids = queries[ROLE].ids
    exact = {s: {q: refs[s - 1][q] for q in ids} for s in SNAPSHOTS}
    expected = {(b, m, s) for b in ('epzero', 'faiss_hnsw') for m in ('incremental', 'shuffled') for s in SEEDS}
    paths = sorted(extension_evidence.rglob('trajectory/report.json'))
    reports, runs, excluded = collect(paths, ROLE, ids, exact, expected=expected)
    # The rebuilt incremental arm must equal the analysed primary runs.
    _, primary, _ = collect(sorted(Path(primary_evidence).rglob('report.json')), ROLE, ids, exact)
    for key, run in runs.items():
        if key[1] == 'incremental' and run != primary[key]:
            raise ValueError(f'extension incremental run differs from the analysed primary run: {key}')
    sweep = {}
    for path in sorted(extension_evidence.rglob('sweep/snapshot-*/report.json')):
        report = json.loads(path.read_text())
        mode = json.loads((path.parents[2] / 'inputs.json').read_text())['mode']
        if report['status'] != 'completed':
            raise ValueError(f'incomplete sweep: {path}')
        for ef, item in report['runs'].items():
            run_path = path.parent / item['file']
            if checksum(run_path) != item['sha256']:
                raise ValueError(f'sweep run checksum mismatch: {run_path}')
            run = read_run(run_path)
            recall = ann_recall(run, exact[report['snapshot']])
            if any(recall[q] != item['ann_recall_per_query'][q] for q in ids):
                raise ValueError(f'sweep recall recount differs: {run_path}')
            sweep[(report['backend'], mode, report['seed'], report['snapshot'], int(ef))] = run
    for (backend, mode, seed, step, ef), run in sweep.items():
        if mode == 'incremental' and ef == 200 and run != primary[(backend, 'incremental', seed, step, 'frozen')]:
            raise ValueError('incremental ef-200 sweep run differs from the primary fixed-ef run')
    qrels, _ = load_test_qrels(release)
    judged = qrels[PRIMARY_FAMILY]
    population = panel(judged, ids)
    base = {s: score(exact[s], judged[s], population) for s in SNAPSHOTS}

    def gap_matrix(lookup, step):
        cols = [score(lookup(seed), judged[step], population) for seed in SEEDS]
        return np.array([[c[q]['nDCG@10'] - base[step][q]['nDCG@10'] for c in cols] for q in population])

    curve = []
    efs = sorted({k[4] for k in sweep})
    for backend in ('epzero', 'faiss_hnsw'):
        for mode in ('incremental', 'shuffled'):
            for step in SNAPSHOTS:
                for ef in efs:
                    gaps = gap_matrix(lambda seed: sweep[(backend, mode, seed, step, ef)], step)
                    recall = np.mean([np.mean(list(ann_recall({q: sweep[(backend, mode, seed, step, ef)][q] for q in population},
                                                              {q: exact[step][q] for q in population}).values())) for seed in SEEDS])
                    curve.append({'backend': backend, 'mode': mode, 'snapshot': step, 'ef': ef, 'recall_mean': float(recall),
                                  **{k: v for k, v in _summary(_estimate(population, list(SEEDS), gaps)).items()
                                     if k in ('mean_difference', 'percentile_interval')}})
    shuffled = {}
    for backend in ('epzero', 'faiss_hnsw'):
        for regime in ('frozen', 'matched_recall'):
            gaps = {s: gap_matrix(lambda seed: runs[(backend, 'shuffled', seed, s, regime)], s) for s in SNAPSHOTS}
            shuffled[f'{backend}/{regime}'] = {
                'gap_by_snapshot': {s: _summary(_estimate(population, list(SEEDS), gaps[s])) for s in SNAPSHOTS},
                'gap_change_D': _summary(_estimate(population, list(SEEDS), gaps[3] - gaps[1])),
                'calibrated_ef': {s: [reports[(backend, 'shuffled', seed)]['snapshots'][s - 1]['calibration']['effort'] for seed in SEEDS]
                                  for s in SNAPSHOTS}}
    first = cohorts(dense)
    ratios = [cohort_ratio(runs, exact, first, (b, m, 'frozen'), s, draws=draws)
              for b in ('epzero', 'faiss_hnsw') for m in ('incremental', 'shuffled') for s in (2, 3)]
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'completed', 'kind': 'EXPLORATORY held-out extension (post-freeze)',
              'source_sha256': checksum(__file__), 'inputs': loaded, 'panel_count': len(population),
              'excluded_attempts': excluded, 'incremental_reproduces_primary': True,
              'sweep_runs': len(sweep), 'curve': curve, 'shuffled': shuffled, 'cohort_ratios': ratios}
    atomic_json(output / 'report.json', report)
    return report
