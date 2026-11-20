"""Post-freeze EXPLORATORY: the judging pool of the held-out runs, as one ledger (not part of FINAL_PROTOCOL_V1).

The released LLM grades are a depth-10 pool of the systems submitted to the lab, so documents that
only the rankings of this study return can be unjudged. This module answers, on one basis, how
many such holes there are, and what happens when the grade the organisers gave the same
query-document pair at another snapshot is carried over.

The ledger has one row per (query, document, snapshot) that appears in a top 10 of the pooled
rankings, with the rankings that returned it. Two units are kept apart everywhere:

  occurrence   a (query, document, snapshot) row; a pair retrieved at three snapshots counts three times
  pair         a distinct (query, document); this is the unit a judge is asked about

Rankings are pooled in three nested sets:

  reported     exact search and every primary held-out run, without the all-lists IVF setting
  +all_lists   adds the all-lists IVF setting (exhaustive probing)
  +extension   adds the exploratory shuffled-order runs and the ef sweep

Everything here describes existing, verified evidence. No new retrieval is run and nothing is judged.
The ledger holds query and document identifiers, so it is written beside the report and never committed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .evaluation import ann_recall, evaluate, read_run
from .heldout import ROLE, load_heldout
from .heldout_analysis import PRIMARY_ARMS, SEEDS, SNAPSHOTS, _estimate, _summary, collect, load_test_qrels, panel
from .release import checksum
from .storage import atomic_json

DEPTH = 10
FAMILY = 'gpt-oss-120b'
SETS = ('reported', '+all_lists', '+extension')          # nested: each includes the ones before it
SCORED = {**PRIMARY_ARMS, 'faiss_hnsw_ef200': ('faiss_hnsw', 'incremental', 'frozen'),
          'epzero_hnsw_ef200': ('epzero', 'incremental', 'frozen')}


def load(inputs, evidence, extension_evidence):
    """Exact references, primary runs and extension runs, each verified against its checksum and recall."""
    queries, refs, _, loaded = load_heldout(Path(inputs))
    ids = queries[ROLE].ids
    exact = {s: {q: refs[s - 1][q] for q in ids} for s in SNAPSHOTS}
    _, primary, _ = collect(sorted(Path(evidence).rglob('report.json')), ROLE, ids, exact)
    extension = {}
    if extension_evidence is not None:
        root = Path(extension_evidence)
        expected = {(b, m, s) for b in ('epzero', 'faiss_hnsw') for m in ('incremental', 'shuffled') for s in SEEDS}
        _, runs, _ = collect(sorted(root.rglob('trajectory/report.json')), ROLE, ids, exact, expected=expected)
        extension.update({('trajectory', *key): run for key, run in runs.items() if key[1] == 'shuffled'})
        for path in sorted(root.rglob('sweep/snapshot-*/report.json')):
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
                extension[('sweep', report['backend'], mode, report['seed'], report['snapshot'], f'ef{ef}')] = run
    return ids, exact, primary, extension, loaded


def ledger(ids, exact, primary, extension):
    """{(query, document, snapshot): {'set': first pooled set that returns it, 'rank': best rank, 'sources': {...}}}."""
    rows = {}

    def add(run, step, level, source):
        for q in ids:
            for rank, (doc, _) in enumerate(run.get(q, [])[:DEPTH], 1):
                row = rows.setdefault((q, doc, step), {'set': level, 'rank': rank, 'sources': set()})
                row['set'] = min(row['set'], level)
                row['rank'] = min(row['rank'], rank)
                row['sources'].add(source)

    for step in SNAPSHOTS:
        add(exact[step], step, 0, 'exact')
    for (backend, mode, seed, step, regime), run in primary.items():
        add(run, step, 1 if regime == 'all_lists' else 0, f'{backend}/{mode}/{regime}/seed{seed}')
    for (kind, backend, mode, seed, step, regime), run in extension.items():
        add(run, step, 2, f'{kind}:{backend}/{mode}/{regime}/seed{seed}')
    return rows


def counts(rows, judged, queries, level):
    """Occurrences and distinct pairs in the pool, split by whether the organisers judged them."""
    queries = set(queries)
    ever = {(q, d) for s in SNAPSHOTS for q, docs in judged[s].items() for d in docs}
    keys = [key for key, row in rows.items() if key[0] in queries and row['set'] <= level]
    holes = [(q, d, s) for q, d, s in keys if d not in judged[s].get(q, {})]
    pairs = {(q, d) for q, d, _ in keys}
    hole_pairs = {(q, d) for q, d, _ in holes}
    return {
        'queries': len(queries),
        'occurrences': {
            'total': len(keys), 'judged_at_that_snapshot': len(keys) - len(holes), 'unjudged_at_that_snapshot': len(holes),
            'unjudged_fillable_by_carry_over': sum((q, d) in ever for q, d, _ in holes),
            'unjudged_never_judged': sum((q, d) not in ever for q, d, _ in holes),
            'by_snapshot': {str(s): {'total': sum(k[2] == s for k in keys),
                                     'unjudged': sum(k[2] == s for k in holes),
                                     'fillable_by_carry_over': sum(k[2] == s and k[:2] in ever for k in holes),
                                     'never_judged': sum(k[2] == s and k[:2] not in ever for k in holes)}
                            for s in SNAPSHOTS}},
        'pairs': {
            'total': len(pairs), 'judged_at_some_snapshot': len(pairs & ever), 'never_judged': len(pairs - ever),
            'with_an_unjudged_occurrence': len(hole_pairs),
            'with_an_unjudged_occurrence_fillable_by_carry_over': len(hole_pairs & ever),
            'with_an_unjudged_occurrence_never_judged': len(hole_pairs - ever)}}


def carried_labels(rows, judged, ids, level, first=None):
    """Official grades plus the grade the organisers gave the same pair at another snapshot.

    Pool-restricted (first is None): a grade is carried only to occurrences in the pooled set `level`.
    Full (first maps document -> arrival snapshot): a grade applies at every snapshot where the
    document exists, whoever retrieved it; this is what pair-level grades imply.
    """
    union = {}
    for s in SNAPSHOTS:
        for q, docs in judged[s].items():
            union.setdefault(q, {}).update(docs)
    out = {s: {q: dict(judged[s].get(q, {})) for q in ids} for s in SNAPSHOTS}
    if first is None:
        for (q, d, s), row in rows.items():
            if row['set'] <= level and d in union.get(q, {}):
                out[s][q].setdefault(d, union[q][d])
    else:
        for q in ids:
            for d, grade in union.get(q, {}).items():
                for s in SNAPSHOTS:
                    if first[d] <= s:
                        out[s][q].setdefault(d, grade)
    added = {str(s): sum(len(out[s][q]) - len(judged[s].get(q, {})) for q in ids) for s in SNAPSHOTS}
    return out, added


def gaps(primary, exact, labels, population):
    """ANN-minus-exact nDCG@10 by snapshot and its change from snapshot 1 to 3 (seeds averaged within query)."""
    def ndcg(run, step):
        scores = evaluate(run, labels[step], population, population='explicit')['per_query']
        return np.array([scores[q]['nDCG@10'] for q in population])

    def judged_share(run, step):
        return float(np.mean([np.mean([d in labels[step].get(q, {}) for d, _ in run[q][:DEPTH]]) for q in population]))

    base = {s: ndcg(exact[s], s) for s in SNAPSHOTS}
    out = {'exact_ndcg': [float(base[s].mean()) for s in SNAPSHOTS],
           'exact_judged_at_10': [judged_share(exact[s], s) for s in SNAPSHOTS], 'arms': {}}
    for name, (backend, mode, regime) in SCORED.items():
        matrix = {s: np.stack([ndcg(primary[(backend, mode, seed, s, regime)], s) - base[s] for seed in SEEDS], axis=1)
                  for s in SNAPSHOTS}
        change = _summary(_estimate(population, list(SEEDS), matrix[3] - matrix[1]))
        out['arms'][name] = {
            'gap_by_snapshot': [float(matrix[s].mean()) for s in SNAPSHOTS],
            'change_s3_minus_s1': change['mean_difference'], 'change_interval_95': change['percentile_interval'],
            'judged_at_10': [float(np.mean([judged_share(primary[(backend, mode, seed, s, regime)], s) for seed in SEEDS]))
                             for s in SNAPSHOTS]}
    return out


def weighted_kappa(confusion):
    """Cohen's kappa with linear weights for a square confusion matrix of ordered grades."""
    confusion = np.asarray(confusion, dtype=np.float64)
    size = confusion.shape[0]
    weight = 1 - np.abs(np.subtract.outer(np.arange(size), np.arange(size))) / (size - 1)
    total = confusion.sum()
    observed = (weight * confusion).sum() / total
    expected = (weight * np.outer(confusion.sum(axis=1), confusion.sum(axis=0))).sum() / total ** 2
    if expected == 1:                                      # every pair in one cell: kappa is undefined
        return float('nan')
    return float((observed - expected) / (1 - expected))


def label_agreement(qrels, exact, ids, *, reference='gpt-oss-120b', other='gpt-oss-20b', draws=2000, seed=20261121):
    """How well the two released LLM label sets agree with each other, pair by pair.

    This is a reference point for validating a locally run judge: it is the agreement that a
    different model reached with the reference labels on the organisers' own pipeline.
    """
    def pairs(family):
        out = {}
        for s in SNAPSHOTS:
            for q, docs in qrels[family][s].items():
                for d, grade in docs.items():
                    if out.setdefault((q, d), grade) != grade:
                        raise ValueError(f'conflicting grades for one pair in {family}')
        return out

    a, b = pairs(reference), pairs(other)
    if a.keys() != b.keys():
        raise ValueError('the two label sets grade different pairs')
    keys = sorted(a)
    left, right = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    grades = int(max(left.max(), right.max())) + 1

    def measures(rows):
        confusion = np.zeros((grades, grades), dtype=np.int64)
        np.add.at(confusion, (left[rows], right[rows]), 1)
        return confusion, float(np.mean((left[rows] > 0) == (right[rows] > 0))), weighted_kappa(confusion)

    confusion, binary, kappa = measures(np.arange(len(keys)))
    by_query = {}
    for i, (q, _) in enumerate(keys):
        by_query.setdefault(q, []).append(i)
    members = [np.array(rows) for rows in by_query.values()]
    rng = np.random.default_rng(seed)
    boot = np.array([measures(np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))]))[1:]
                     for _ in range(draws)])

    def dcg(gains):
        return sum(g / np.log2(i + 2) for i, g in enumerate(gains))

    lists = {}
    for s in SNAPSHOTS:
        scores = []
        for q in ids:
            docs = [d for d, _ in exact[s][q][:DEPTH]]
            if not all(d in qrels[reference][s].get(q, {}) for d in docs):
                continue
            row = []
            for grades_of in (a, b):                       # grades are per pair, so the pair tables apply
                gains = [max(0, grades_of[(q, d)]) for d in docs]
                ideal = dcg(sorted(gains, reverse=True))
                row.append(dcg(gains) / ideal if ideal else 0.0)
            scores.append(row)
        scores = np.array(scores, dtype=np.float64).reshape(-1, 2)
        if not len(scores):
            lists[str(s)] = {'complete_lists': 0}
            continue
        lists[str(s)] = {'complete_lists': len(scores), 'ndcg_reference': float(scores[:, 0].mean()),
                         'ndcg_other': float(scores[:, 1].mean()),
                         'difference': float(scores[:, 1].mean() - scores[:, 0].mean()),
                         'mean_absolute_difference_per_query': float(np.abs(scores[:, 1] - scores[:, 0]).mean())}
    return {'reference': reference, 'other': other, 'pairs': len(keys), 'queries': len(members),
            'confusion_reference_rows_other_columns': confusion.tolist(),
            'exact_agreement': float(np.mean(left == right)),
            'binary_agreement': binary, 'binary_agreement_interval_95': np.quantile(boot[:, 0], [.025, .975]).tolist(),
            'linear_weighted_kappa': kappa, 'linear_weighted_kappa_interval_95': np.quantile(boot[:, 1], [.025, .975]).tolist(),
            'bootstrap': {'unit': 'query', 'draws': draws, 'seed': seed},
            # Exact-search top 10 fully graded by the reference; ideal DCG from the same ten documents.
            'exact_top10_complete_lists': lists}


def _added(labels, judged, queries):
    return {str(s): sum(len(labels[s][q]) - len(judged[s].get(q, {})) for q in queries) for s in SNAPSHOTS}


def analyze(inputs, release, evidence, extension_evidence, output, *, dense=None):
    output = Path(output)
    ids, exact, primary, extension, loaded = load(inputs, evidence, extension_evidence)
    qrels, identity = load_test_qrels(release)
    judged = qrels[FAMILY]
    population = panel(judged, ids)
    rows = ledger(ids, exact, primary, extension)
    output.mkdir(parents=True, exist_ok=False)
    with open(output / 'ledger.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['query', 'document', 'snapshot', 'pooled_set', 'best_rank', 'judged_at_snapshot',
                         'judged_at_other_snapshot', 'sources'])
        for (q, d, s), row in sorted(rows.items()):
            here = d in judged[s].get(q, {})
            other = any(d in judged[t].get(q, {}) for t in SNAPSHOTS if t != s)
            writer.writerow([q, d, s, SETS[row['set']], row['rank'], int(here), int(other), ';'.join(sorted(row['sources']))])
    report = {'status': 'completed', 'kind': 'EXPLORATORY post-freeze judging-pool ledger and carried grades',
              'source_sha256': checksum(__file__), 'inputs': loaded, 'judgment_sha256': identity, 'family': FAMILY,
              'depth': DEPTH, 'ledger_rows': len(rows), 'ledger_sha256': checksum(output / 'ledger.csv'),
              'counts': {name: {'all_test_queries': counts(rows, judged, ids, level),
                                'primary_panel': counts(rows, judged, population, level)}
                         for level, name in enumerate(SETS) if level < 2 or extension},
              'released_label_agreement': label_agreement(qrels, exact, ids),
              'gaps': {'official': gaps(primary, exact, judged, population)}}
    for level, name in ((0, 'carried_within_reported_pool'), (1, 'carried_within_pool_with_all_lists')):
        labels, added = carried_labels(rows, judged, ids, level)
        report['gaps'][name] = {'labels_added_all_test_queries': added,
                                'labels_added_primary_panel': _added(labels, judged, population),
                                **gaps(primary, exact, labels, population)}
    if dense is not None:
        from .heldout_exploratory import cohorts
        labels, added = carried_labels(rows, judged, ids, 0, first=cohorts(dense))
        report['gaps']['carried_wherever_the_document_exists'] = {
            'labels_added_all_test_queries': added, 'labels_added_primary_panel': _added(labels, judged, population),
            **gaps(primary, exact, labels, population)}
    atomic_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('inputs', 'release', 'evidence', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--extension-evidence', type=Path)
    parser.add_argument('--dense', type=Path, help='prepared dense snapshots; enables the full carry-over')
    a = parser.parse_args()
    r = analyze(a.inputs, a.release, a.evidence, a.extension_evidence, a.output, dense=a.dense)
    print(json.dumps({'counts': r['counts'], 'gaps': r['gaps']}, indent=2))


if __name__ == '__main__':
    main()
