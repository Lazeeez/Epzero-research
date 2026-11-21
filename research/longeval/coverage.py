"""EXPLORATORY follow-up: the judge's validation sample, and the effect of judgment coverage on the gap.

Not part of FINAL_PROTOCOL_V1. Definitions follow docs/protocol/JUDGE_VALIDATION_THRESHOLDS.md and the
extension protocol draft. This module draws the pairs to grade, grades them with the pinned judge
configuration, and computes two things from the local grades:

  validation   does the local judge agree with the released grades (three thresholds)?
  coverage     how much of the approximate-minus-exact gap comes from documents that the release
               never graded? The same local grades are scored twice: with every grade visible
               ("full") and with the grades hidden wherever the release has none ("masked").

The judge itself (prompt, parser, request) is `judge.py` and is not changed here.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from . import judge
from .release import checksum, read_queries
from .storage import atomic_json

SEED = 20261121
VALIDATION_PAIRS = 1000
THRESHOLDS = {'binary_agreement': .90, 'linear_weighted_kappa': .80, 'ndcg_difference': .02}
DEPTH = 10
SNAPSHOTS = (1, 2, 3)


# ------------------------------------------------------------------------------------------ sampling
def validation_sample(pairs, excluded, *, size=VALIDATION_PAIRS, seed=SEED):
    """Officially graded pairs for validation: every grade-3 pair, and the rest split equally over
    (grade 0-2) x (first graded snapshot), chosen by a keyed hash. Returns rows with the stratum,
    its size and the selection probability of each pair."""
    strata = {}
    for (query, doc), (grade, step) in pairs.items():
        if (query, doc) not in excluded:
            key = hashlib.sha256(f'judge-validation-v1:{seed}:{query}:{doc}'.encode()).digest()
            strata.setdefault((grade, step), []).append((key, query, doc))
    top = sorted(stratum for stratum in strata if stratum[0] >= 3)
    rest = sorted(stratum for stratum in strata if stratum[0] < 3)
    certain = sum(len(strata[stratum]) for stratum in top)
    share, extra = divmod(size - certain, len(rest))          # equal shares; the first strata take the remainder
    wanted = {stratum: share + (index < extra) for index, stratum in enumerate(rest)}
    wanted.update({stratum: len(strata[stratum]) for stratum in top})
    rows = []
    for stratum in sorted(strata):
        members = sorted(strata[stratum])
        if len(members) < wanted[stratum]:
            raise ValueError(f'stratum {stratum} has only {len(members)} eligible pairs')
        for _, query, doc in members[:wanted[stratum]]:
            rows.append({'query_id': query, 'doc_id': doc, 'official_grade': stratum[0], 'first_graded_snapshot': stratum[1],
                         'stratum_pairs': len(members), 'selection_probability': wanted[stratum] / len(members)})
    return rows


def prepare(release, ledger, pilot_inputs, output, *, pool_sets=('reported',), size=VALIDATION_PAIRS, seed=SEED):
    """Write the pairs to grade: the validation sample and the study pool, each distinct pair once.

    `pool_sets` names the pooled sets of the ledger that contribute ('reported', '+all_lists', '+extension').
    The pilot's prompt-development pairs are excluded from the validation sample.
    """
    from .heldout_analysis import load_test_qrels
    release, ledger, output = Path(release), Path(ledger), Path(output)
    queries = read_queries(release / 'files' / judge.TEST_FILE)
    if checksum(release / 'files' / judge.TEST_FILE) != judge.TEST_SHA256:
        raise ValueError('test query release identity mismatch')
    qrels, identity = load_test_qrels(release / 'files')
    official = {key: value for key, value in judge.official_pairs(qrels[judge.FAMILY]).items() if key[0] in queries}
    pilot = {(row['query_id'], row['doc_id']) for row in judge._read_jsonl(Path(pilot_inputs) / 'pairs.jsonl')}
    with ledger.open(newline='') as handle:
        pool = sorted({(row['query'], row['document']) for row in csv.DictReader(handle) if row['pooled_set'] in pool_sets})
    sample = validation_sample(official, pilot, size=size, seed=seed)
    weight = {(row['query_id'], row['doc_id']): row for row in sample}
    everything = sorted(set(weight) | set(pool))
    random.Random(seed).shuffle(everything)
    texts = judge.document_texts(release / 'documents', {doc for _, doc in everything})
    in_pool = set(pool)
    output.mkdir(parents=True, exist_ok=False)
    rows, key = [], []
    for number, (query, doc) in enumerate(everything, 1):
        rows.append({'pair': number, 'query_id': query, 'doc_id': doc, 'query': queries[query], 'text': texts[doc]})
        item = weight.get((query, doc))
        key.append({'pair': number, 'validation': item is not None, 'pool': (query, doc) in in_pool,
                    'official_grade': official[(query, doc)][0] if (query, doc) in official else None,
                    'selection_probability': item['selection_probability'] if item else None,
                    'stratum_pairs': item['stratum_pairs'] if item else None,
                    'first_graded_snapshot': item['first_graded_snapshot'] if item else None})
    judge._write_jsonl(output / 'pairs.jsonl', rows)
    judge._write_jsonl(output / 'key.jsonl', key)
    contract = {'status': 'prepared', 'purpose': 'judge validation sample and study pool; one grade per distinct pair',
                'pairs': len(rows), 'validation_pairs': len(sample), 'pool_pairs': len(pool),
                'pairs_in_both': len(set(weight) & in_pool), 'pool_sets': list(pool_sets), 'seed': seed,
                'pilot_pairs_excluded_from_validation': len(pilot),
                'prompt_sha256': hashlib.sha256(judge.prompt('{query}', '{document}').encode()).hexdigest(),
                'judgment_sha256': {f'{judge.FAMILY}/snapshot-{s}': identity[f'{judge.FAMILY}/snapshot-{s}'] for s in SNAPSHOTS},
                'ledger_sha256': checksum(ledger), 'pilot_inputs_sha256': checksum(Path(pilot_inputs) / 'inputs.json'),
                'source_sha256': checksum(__file__),
                'files': {name: checksum(output / name) for name in ('pairs.jsonl', 'key.jsonl')}}
    atomic_json(output / 'inputs.json', contract)
    return contract


def grade(inputs, output, *, base_url, params, model=judge.MODEL_ID, workers=16):
    """Grade every pair once under the pinned request parameters; interrupted runs resume."""
    inputs, output = Path(inputs), Path(output)
    contract = json.loads((inputs / 'inputs.json').read_text())
    for name, digest in contract['files'].items():
        if checksum(inputs / name) != digest:
            raise ValueError(f'input checksum mismatch: {name}')
    rows = judge._read_jsonl(inputs / 'pairs.jsonl')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'responses.jsonl'
    done = {row['pair'] for row in judge._read_jsonl(path) if row['status'] == 'ok'} if path.exists() else set()
    todo = [row for row in rows if row['pair'] not in done]
    with path.open('a', encoding='utf-8', newline='\n') as stream, \
            concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(lambda row: judge.ask(base_url, model, row, params), todo):
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
    record = {'status': 'completed', 'model': model, 'params': params, 'max_tokens': judge.MAX_TOKENS,
              'requested': len(todo), 'already_done': len(done), 'inputs_sha256': checksum(inputs / 'inputs.json'),
              'source_sha256': checksum(__file__), 'judge_source_sha256': checksum(judge.__file__)}
    atomic_json(output / 'grading.json', record)
    return record


def local_grades(inputs, output):
    """{(query, document): grade} from a grading run; a pair without a parsed answer is left out and counted."""
    rows = {row['pair']: row for row in judge._read_jsonl(Path(inputs) / 'pairs.jsonl')}
    grades, unparsed = {}, 0
    for row in judge._read_jsonl(Path(output) / 'responses.jsonl'):
        content = judge._content(row)
        value, valid = judge.parse_grade(content) if content is not None else (0, 0)
        if not valid:
            unparsed += 1
            continue
        grades[(rows[row['pair']]['query_id'], rows[row['pair']]['doc_id'])] = value
    return grades, unparsed


# ------------------------------------------------------------------------------------------ validation
def _weighted(local, official, weights, size=4):
    confusion = np.zeros((size, size))
    np.add.at(confusion, (official, local), weights)
    from .heldout_pool import weighted_kappa
    binary = float(np.sum(weights * ((official > 0) == (local > 0))) / np.sum(weights))
    return binary, weighted_kappa(confusion)


def validation(key, pairs, grades, *, draws=10_000, seed=SEED):
    """Measures 1 and 2 of the thresholds record: population-weighted binary agreement and
    linear-weighted kappa on the validation sample, with a query-clustered bootstrap."""
    rows = [(item, pairs[item['pair']]) for item in key if item['validation']]
    graded = [(item, pair) for item, pair in rows if (pair['query_id'], pair['doc_id']) in grades]
    official = np.array([item['official_grade'] for item, _ in graded])
    local = np.array([grades[(pair['query_id'], pair['doc_id'])] for _, pair in graded])
    weights = np.array([1 / item['selection_probability'] for item, _ in graded])
    clusters = {}
    for index, (_, pair) in enumerate(graded):
        clusters.setdefault(pair['query_id'], []).append(index)
    groups = [np.array(members) for members in clusters.values()]
    rng = np.random.default_rng(seed)
    boot = np.array([_weighted(local[rows_], official[rows_], weights[rows_])
                     for rows_ in (np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
                                   for _ in range(draws))])
    binary, kappa = _weighted(local, official, weights)
    unweighted = np.zeros((4, 4), dtype=np.int64)
    np.add.at(unweighted, (official, local), 1)
    out = {'validation_pairs': len(rows), 'graded': len(graded), 'queries': len(groups),
           'binary_agreement': binary, 'binary_agreement_interval_95': np.quantile(boot[:, 0], [.025, .975]).tolist(),
           'linear_weighted_kappa': kappa, 'linear_weighted_kappa_interval_95': np.nanquantile(boot[:, 1], [.025, .975]).tolist(),
           'agreement_by_official_grade': {str(g): float(np.mean(local[official == g] == g)) for g in sorted(set(official.tolist()))},
           'unweighted_confusion_official_rows_local_columns': unweighted.tolist(),
           'bootstrap': {'unit': 'query', 'draws': draws, 'seed': seed}}
    for name in ('binary_agreement', 'linear_weighted_kappa'):
        met = out[name] >= THRESHOLDS[name]
        out[name + '_threshold'] = {'required': THRESHOLDS[name], 'met': bool(met),
                                    'interval_includes_threshold': bool(met and out[name + '_interval_95'][0] < THRESHOLDS[name])}
    return out


def _dcg(gains):
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def list_agreement(exact, official, grades, queries):
    """Measure 3: mean nDCG@10 of complete exact top-10 lists under local and under official grades.
    A list is complete when all ten documents have an official grade at that snapshot and a local
    grade; its ideal DCG is taken from the same ten documents under each assessor."""
    out = {}
    for step in SNAPSHOTS:
        scores = []
        for query in queries:
            docs = [doc for doc, _ in exact[step][query][:DEPTH]]
            judged = official[step].get(query, {})
            if not all(doc in judged and (query, doc) in grades for doc in docs):
                continue
            row = []
            for gains in ([max(0, judged[doc]) for doc in docs], [grades[(query, doc)] for doc in docs]):
                ideal = _dcg(sorted(gains, reverse=True))
                row.append(_dcg(gains) / ideal if ideal else 0.0)
            scores.append(row)
        scores = np.array(scores, dtype=np.float64).reshape(-1, 2)
        if not len(scores):
            out[str(step)] = {'complete_lists': 0}
            continue
        difference = float(scores[:, 1].mean() - scores[:, 0].mean())
        out[str(step)] = {'complete_lists': len(scores), 'ndcg_official': float(scores[:, 0].mean()),
                          'ndcg_local': float(scores[:, 1].mean()), 'difference': difference,
                          'met': bool(abs(difference) <= THRESHOLDS['ndcg_difference'])}
    measured = [item for item in out.values() if item['complete_lists']]
    return {'by_snapshot': out, 'required': THRESHOLDS['ndcg_difference'],
            'met': bool(measured) and all(item['met'] for item in measured)}


# ------------------------------------------------------------------------------------------ coverage
def pool_documents(primary, exact, arms, queries):
    """{snapshot: {query: documents in the top 10 of exact search or of a contributing run}}."""
    wanted = set(arms.values())
    out = {step: {query: {doc for doc, _ in exact[step][query][:DEPTH]} for query in queries} for step in SNAPSHOTS}
    for (backend, mode, seed, step, regime), run in primary.items():
        if (backend, mode, regime) in wanted:
            for query in queries:
                out[step][query].update(doc for doc, _ in run[query][:DEPTH])
    return out


def label_sets(official, grades, pool):
    """The grade tables of the five estimates, each {snapshot: {query: {document: grade}}}."""
    full = {s: {q: {d: grades[(q, d)] for d in docs if (q, d) in grades} for q, docs in pool[s].items()} for s in SNAPSHOTS}
    masked = {s: {q: {d: g for d, g in docs.items() if d in official[s].get(q, {})} for q, docs in full[s].items()} for s in SNAPSHOTS}
    filled = {s: {q: {**{d: g for d, g in full[s][q].items()}, **official[s].get(q, {})} for q in pool[s]} for s in SNAPSHOTS}
    plain = {s: {q: dict(official[s].get(q, {})) for q in pool[s]} for s in SNAPSHOTS}
    return {'official': plain, 'official_plus_filled': filled, 'local_masked': masked, 'local_full': full}


def coverage(primary, exact, official, grades, arms, population):
    """The gap of each arm under each estimate, and the primary coverage estimate C = G(full) - G(masked)."""
    from .evaluation import evaluate
    from .heldout_analysis import SEEDS, _estimate, _summary
    pool = pool_documents(primary, exact, arms, population)
    labels = label_sets(official, grades, pool)

    def gaps(table, step, backend, mode, regime, *, condensed=False):
        def scores(run):
            if condensed:
                run = {q: [(d, s) for d, s in run[q] if d in table[step].get(q, {})] for q in population}
            return evaluate(run, table[step], population, population='explicit')['per_query']
        base = scores(exact[step])
        columns = [scores(primary[(backend, mode, seed, step, regime)]) for seed in SEEDS]
        return np.array([[column[q]['nDCG@10'] - base[q]['nDCG@10'] for column in columns] for q in population])

    out = {'queries': len(population),
           'pool_pairs_without_a_local_grade': sum((q, d) not in grades for s in SNAPSHOTS for q, docs in pool[s].items() for d in docs),
           'arms': {}}
    for name, (backend, mode, regime) in arms.items():
        matrix = {estimate: {s: gaps(table, s, backend, mode, regime) for s in SNAPSHOTS} for estimate, table in labels.items()}
        matrix['condensed'] = {s: gaps(labels['official'], s, backend, mode, regime, condensed=True) for s in SNAPSHOTS}
        item = {'gap_by_snapshot': {estimate: [float(values[s].mean()) for s in SNAPSHOTS] for estimate, values in matrix.items()},
                'change_s3_minus_s1': {}}
        for estimate, values in matrix.items():
            change = _summary(_estimate(population, list(SEEDS), values[3] - values[1]))
            item['change_s3_minus_s1'][estimate] = {'mean': change['mean_difference'], 'interval_95': change['percentile_interval']}
        effect = {s: matrix['local_full'][s] - matrix['local_masked'][s] for s in SNAPSHOTS}
        item['coverage_effect'] = {}
        for s in SNAPSHOTS:
            summary = _summary(_estimate(population, list(SEEDS), effect[s]))
            item['coverage_effect'][str(s)] = {'mean': summary['mean_difference'], 'interval_95': summary['percentile_interval']}
        summary = _summary(_estimate(population, list(SEEDS), effect[3] - effect[1]))
        item['coverage_effect']['s3_minus_s1'] = {'mean': summary['mean_difference'], 'interval_95': summary['percentile_interval']}
        zero = {estimate: [int(sum(not any(g > 0 for g in table[s].get(q, {}).values()) for q in population)) for s in SNAPSHOTS]
                for estimate, table in labels.items()}
        item['queries_without_a_positive_grade'] = zero
        out['arms'][name] = item
    return out


def analyze(grading_inputs, grading_output, heldout_inputs, release, evidence, output):
    """Validation (three thresholds) and coverage from one grading run."""
    from .heldout_analysis import PRIMARY_ARMS, load_test_qrels, panel
    from .heldout_pool import load
    grading_inputs, output = Path(grading_inputs), Path(output)
    key = judge._read_jsonl(grading_inputs / 'key.jsonl')
    pairs = {row['pair']: row for row in judge._read_jsonl(grading_inputs / 'pairs.jsonl')}
    grades, unparsed = local_grades(grading_inputs, grading_output)
    ids, exact, primary, _, loaded = load(heldout_inputs, evidence, None)
    qrels, identity = load_test_qrels(Path(release) / 'files')
    official = qrels[judge.FAMILY]
    population = panel(official, ids)
    report = {'status': 'completed', 'kind': 'EXPLORATORY follow-up: judge validation and judgment coverage',
              'pairs': len(pairs), 'graded': len(grades), 'unparsed': unparsed,
              'thresholds': THRESHOLDS, 'validation': validation(key, pairs, grades),
              'list_agreement': list_agreement(exact, official, grades, ids),
              'coverage': coverage(primary, exact, official, grades, PRIMARY_ARMS, population),
              'inputs': loaded, 'grading': json.loads((Path(grading_output) / 'grading.json').read_text()),
              'source_sha256': checksum(__file__)}
    report['compatible_with_the_official_assessor'] = bool(
        report['validation']['binary_agreement_threshold']['met'] and report['validation']['linear_weighted_kappa_threshold']['met']
        and report['list_agreement']['met'])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='draw the validation sample and write the pairs to grade')
    for name in ('release', 'ledger', 'pilot_inputs', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--pool-sets', nargs='+', default=['reported'], choices=('reported', '+all_lists', '+extension'))
    p = sub.add_parser('grade', help='grade every pair once with the pinned request parameters')
    p.add_argument('inputs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--base-url', required=True)
    p.add_argument('--temperature', type=float, help='omit to send no sampling parameter, as the organisers did')
    p.add_argument('--workers', type=int, default=16)
    p = sub.add_parser('analyze', help='validation and coverage from a grading run')
    for name in ('grading_inputs', 'grading_output', 'heldout_inputs', 'release', 'evidence', 'output'):
        p.add_argument(name, type=Path)
    a = parser.parse_args()
    if a.command == 'prepare':
        r = prepare(a.release, a.ledger, a.pilot_inputs, a.output, pool_sets=tuple(a.pool_sets))
        r = {name: r[name] for name in ('pairs', 'validation_pairs', 'pool_pairs', 'pairs_in_both', 'pool_sets')}
    elif a.command == 'grade':
        params = {} if a.temperature is None else {'temperature': a.temperature}
        r = grade(a.inputs, a.output, base_url=a.base_url, params=params, workers=a.workers)
    else:
        r = analyze(a.grading_inputs, a.grading_output, a.heldout_inputs, a.release, a.evidence, a.output)
        r = {'compatible_with_the_official_assessor': r['compatible_with_the_official_assessor'],
             'validation': {k: r['validation'][k] for k in ('binary_agreement', 'linear_weighted_kappa')},
             'list_agreement': r['list_agreement']['met']}
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
