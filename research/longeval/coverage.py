"""EXPLORATORY follow-up: the judge's validation sample, and the effect of judgment coverage on the gap.

Not part of FINAL_PROTOCOL_V1. Definitions follow docs/protocol/JUDGE_VALIDATION_THRESHOLDS.md and the
extension protocol draft. This module draws the pairs to grade, grades them with the pinned judge
configuration, and computes two things from the local grades:

  validation   does the local judge agree with the released grades (three thresholds)?
  coverage     how much of the approximate-minus-exact gap comes from documents that the release
               never graded? The same local grades are scored twice: with every grade visible
               ("full") and with the grades hidden wherever the release has none ("masked").

Three rules keep a result attributable to one experiment:

  binding      a grading folder belongs to one configuration (inputs, model revision, prompt, parser,
               request parameters, answer cap, server settings). It is recorded before the first
               request, and a run that resumes under another configuration is refused.
  usable       a pair has a grade only if an answer parsed. An answer that does not parse is asked
               again with the same request, at most `RETRIES` more times; every attempt is kept and
               the first parsed answer is the grade. A pair that never parses has no grade. (The
               organisers' pipeline wrote grade 0 in that case; here it is reported as missing.)
  complete     the primary estimates need a usable grade for every validation pair and every pair
               of the declared scoring universe. Otherwise the analysis stops, or runs as a
               diagnostic that is marked as such.

The scoring universe (which documents of a query count at a snapshot) is written when the pairs
are drawn, from the pooled runs that were chosen, and the analysis uses exactly that.

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
RETRIES = 2                           # further attempts for a pair whose answer failed or did not parse
INPUT_FILES = ('pairs.jsonl', 'key.jsonl', 'universe.json')


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


def read_universe(ledger, pool_sets):
    """{snapshot: {query: sorted documents}} of the chosen pooled sets, from the judging-pool ledger."""
    universe = {step: {} for step in SNAPSHOTS}
    with Path(ledger).open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['pooled_set'] in pool_sets:
                universe[int(row['snapshot'])].setdefault(row['query'], set()).add(row['document'])
    return {step: {query: sorted(docs) for query, docs in sorted(queries.items())} for step, queries in universe.items()}


def prepare(release, ledger, pilot_inputs, output, *, pool_sets=('reported',), size=VALIDATION_PAIRS, seed=SEED):
    """Write the pairs to grade: the validation sample and the study pool, each distinct pair once.

    `pool_sets` names the pooled sets of the ledger that contribute ('reported', '+all_lists', '+extension').
    They define both the pairs that are graded and the scoring universe, which is written to
    `universe.json`. The pilot's prompt-development pairs are excluded from the validation sample.
    """
    from .heldout_analysis import load_test_qrels
    release, ledger, output = Path(release), Path(ledger), Path(output)
    queries = read_queries(release / 'files' / judge.TEST_FILE)
    if checksum(release / 'files' / judge.TEST_FILE) != judge.TEST_SHA256:
        raise ValueError('test query release identity mismatch')
    qrels, identity = load_test_qrels(release / 'files')
    official = {key: value for key, value in judge.official_pairs(qrels[judge.FAMILY]).items() if key[0] in queries}
    pilot = {(row['query_id'], row['doc_id']) for row in judge._read_jsonl(Path(pilot_inputs) / 'pairs.jsonl')}
    universe = read_universe(ledger, pool_sets)
    pool = sorted({(query, doc) for step in SNAPSHOTS for query, docs in universe[step].items() for doc in docs})
    if not pool:
        raise ValueError('the chosen pooled sets are empty')
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
    atomic_json(output / 'universe.json', {'pool_sets': list(pool_sets), 'depth': DEPTH,
                                           'snapshots': {str(step): universe[step] for step in SNAPSHOTS}})
    contract = {'status': 'prepared', 'purpose': 'judge validation sample and study pool; one grade per distinct pair',
                'pairs': len(rows), 'validation_pairs': len(sample), 'pool_pairs': len(pool),
                'pairs_in_both': len(set(weight) & in_pool), 'pool_sets': list(pool_sets), 'seed': seed,
                'universe_occurrences': {str(step): sum(len(docs) for docs in universe[step].values()) for step in SNAPSHOTS},
                'pilot_pairs_excluded_from_validation': len(pilot),
                'prompt_sha256': hashlib.sha256(judge.prompt('{query}', '{document}').encode()).hexdigest(),
                'judgment_sha256': {f'{judge.FAMILY}/snapshot-{s}': identity[f'{judge.FAMILY}/snapshot-{s}'] for s in SNAPSHOTS},
                'ledger_sha256': checksum(ledger), 'pilot_inputs_sha256': checksum(Path(pilot_inputs) / 'inputs.json'),
                'source_sha256': checksum(__file__), 'files': {name: checksum(output / name) for name in INPUT_FILES}}
    atomic_json(output / 'inputs.json', contract)
    return contract


def read_inputs(inputs):
    """The input contract, after checking every input file against it."""
    inputs = Path(inputs)
    contract = json.loads((inputs / 'inputs.json').read_text())
    if set(contract['files']) != set(INPUT_FILES):
        raise ValueError('not a prepared grading input folder')
    for name, digest in contract['files'].items():
        if checksum(inputs / name) != digest:
            raise ValueError(f'input checksum mismatch: {name}')
    return contract


# ------------------------------------------------------------------------------------------ grading
def binding(inputs, *, model, params, server_config=None):
    """What a grading folder is bound to: the inputs in their order, and everything that shapes an answer."""
    contract = read_inputs(inputs)
    return {'inputs_sha256': checksum(Path(inputs) / 'inputs.json'), 'pairs_sha256': contract['files']['pairs.jsonl'],
            'model': model, 'model_revision': judge.MODEL_REVISION,
            'prompt_sha256': hashlib.sha256(judge.prompt('{query}', '{document}').encode()).hexdigest(),
            'parser_sha256': hashlib.sha256('\n'.join(judge._PATTERNS).encode()).hexdigest(),
            'request_parameters': params, 'max_tokens': judge.MAX_TOKENS, 'retries': RETRIES,
            'server_config_sha256': checksum(server_config) if server_config is not None else None,
            'judge_source_sha256': checksum(judge.__file__)}


def usable(row):
    """An answer that arrived and parsed."""
    content = judge._content(row)
    return content is not None and judge.parse_grade(content)[1] == 1


def grade(inputs, output, *, base_url, params, model=judge.MODEL_ID, server_config=None, workers=16):
    """Grade every pair under one bound configuration. An interrupted run resumes; a pair without
    a usable answer is asked again, at most `RETRIES` more times, and every attempt is kept."""
    inputs, output = Path(inputs), Path(output)
    bound = binding(inputs, model=model, params=params, server_config=server_config)
    rows = judge._read_jsonl(inputs / 'pairs.jsonl')
    output.mkdir(parents=True, exist_ok=True)
    path, record_path = output / 'responses.jsonl', output / 'binding.json'
    if record_path.exists():
        if json.loads(record_path.read_text()) != bound:
            raise ValueError('this output folder is bound to another grading configuration; use a new folder')
    elif path.exists():
        raise ValueError('responses without a binding record cannot be resumed')
    else:
        atomic_json(record_path, bound)
    attempts, good = {}, set()
    for row in judge._read_jsonl(path) if path.exists() else []:
        attempts[row['pair']] = attempts.get(row['pair'], 0) + 1
        if usable(row):
            good.add(row['pair'])
    rounds = []
    for number in range(1 + RETRIES):
        todo = [row for row in rows if row['pair'] not in good and attempts.get(row['pair'], 0) <= number]
        if not todo:
            continue
        with path.open('a', encoding='utf-8', newline='\n') as stream, \
                concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(lambda row: judge.ask(base_url, model, row, params), todo):
                result['attempt'] = attempts.get(result['pair'], 0) + 1
                attempts[result['pair']] = result['attempt']
                if usable(result):
                    good.add(result['pair'])
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
        rounds.append({'attempt': number + 1, 'requested': len(todo)})
    record = {'status': 'completed', 'binding_sha256': checksum(record_path), 'rounds': rounds, 'pairs': len(rows),
              'pairs_with_a_usable_answer': len(good), 'pairs_without_a_usable_answer': len(rows) - len(good),
              'source_sha256': checksum(__file__)}
    atomic_json(output / 'grading.json', record)
    return record


def local_grades(inputs, output):
    """{(query, document): grade} from a grading run: the first usable answer of each pair."""
    rows = {row['pair']: row for row in judge._read_jsonl(Path(inputs) / 'pairs.jsonl')}
    grades, responses, several = {}, 0, 0
    for row in judge._read_jsonl(Path(output) / 'responses.jsonl'):
        responses += 1
        if not usable(row):
            continue
        key = (rows[row['pair']]['query_id'], rows[row['pair']]['doc_id'])
        if key in grades:
            several += 1                                   # cannot arise from `grade`; counted if files were merged
            continue
        grades[key] = judge.parse_grade(judge._content(row))[0]
    return grades, {'responses': responses, 'pairs_with_a_usable_answer': len(grades),
                    'pairs_without_a_usable_answer': len(rows) - len(grades), 'surplus_usable_answers': several}


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
    out = {'validation_pairs': len(rows), 'graded': len(graded), 'complete': len(graded) == len(rows), 'queries': len(groups),
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
def missing_grades(universe, grades, official, queries):
    """Pairs of the scoring universe without a usable local grade, by snapshot and by whether the
    release graded them there."""
    out = {}
    for step in SNAPSHOTS:
        counts = {'occurrences': 0, 'missing': 0, 'missing_officially_graded': 0, 'missing_officially_ungraded': 0}
        for query in queries:
            for doc in universe[step].get(query, ()):
                counts['occurrences'] += 1
                if (query, doc) not in grades:
                    counts['missing'] += 1
                    counts['missing_officially_graded' if doc in official[step].get(query, {}) else 'missing_officially_ungraded'] += 1
        out[str(step)] = counts
    return out


def label_sets(official, grades, universe):
    """The grade tables of the estimates, each {snapshot: {query: {document: grade}}}."""
    full = {s: {q: {d: grades[(q, d)] for d in docs if (q, d) in grades} for q, docs in universe[s].items()} for s in SNAPSHOTS}
    masked = {s: {q: {d: g for d, g in docs.items() if d in official[s].get(q, {})} for q, docs in full[s].items()} for s in SNAPSHOTS}
    filled = {s: {q: {**full[s][q], **official[s].get(q, {})} for q in universe[s]} for s in SNAPSHOTS}
    plain = {s: {q: dict(official[s].get(q, {})) for q in universe[s]} for s in SNAPSHOTS}
    return {'official': plain, 'official_plus_filled': filled, 'local_masked': masked, 'local_full': full}


def coverage(primary, exact, official, grades, arms, population, universe):
    """The gap of each arm under each estimate, and the primary coverage estimate C = G(full) - G(masked).

    `universe` is {snapshot: {query: documents}}: the documents that count for a query at a snapshot.
    The local estimates take their ideal DCG from those documents under the grades that are visible.
    """
    from .evaluation import evaluate
    from .heldout_analysis import SEEDS, _estimate, _summary
    labels = label_sets(official, grades, universe)

    def gaps(table, step, backend, mode, regime, *, condensed=False):
        def scores(run):
            if condensed:
                run = {q: [(d, s) for d, s in run[q] if d in table[step].get(q, {})] for q in population}
            return evaluate(run, table[step], population, population='explicit')['per_query']
        base = scores(exact[step])
        columns = [scores(primary[(backend, mode, seed, step, regime)]) for seed in SEEDS]
        return np.array([[column[q]['nDCG@10'] - base[q]['nDCG@10'] for column in columns] for q in population])

    missing = missing_grades(universe, grades, official, population)
    out = {'queries': len(population), 'missing_grades': missing,
           'complete': all(item['missing'] == 0 for item in missing.values()), 'arms': {}}
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
        item['queries_without_a_positive_grade'] = {
            estimate: [int(sum(not any(g > 0 for g in table[s].get(q, {}).values()) for q in population)) for s in SNAPSHOTS]
            for estimate, table in labels.items()}
        out['arms'][name] = item
    return out


def require_complete(checked, effect, *, allow_partial=False):
    """True when every validation pair and every pair of the scoring universe has a usable grade.
    Otherwise it raises, or returns False when a diagnostic report was asked for."""
    complete = checked['complete'] and effect['complete']
    if not complete and not allow_partial:
        raise ValueError('grades are missing: %d validation pairs and %s universe occurrences by snapshot; '
                         'rerun the grading, or pass allow_partial for a diagnostic report'
                         % (checked['validation_pairs'] - checked['graded'],
                            {step: item['missing'] for step, item in effect['missing_grades'].items()}))
    return complete


def analyze(grading_inputs, grading_output, heldout_inputs, release, evidence, output, *, allow_partial=False):
    """Validation (three thresholds) and coverage from one grading run.

    Stops if a validation pair or a pair of the scoring universe has no usable grade, unless
    `allow_partial` is set; a partial report is marked as a diagnostic and decides nothing.
    """
    from .heldout_analysis import PRIMARY_ARMS, load_test_qrels, panel
    from .heldout_pool import load
    grading_inputs, grading_output, output = Path(grading_inputs), Path(grading_output), Path(output)
    contract = read_inputs(grading_inputs)
    bound = json.loads((grading_output / 'binding.json').read_text())
    if bound['inputs_sha256'] != checksum(grading_inputs / 'inputs.json'):
        raise ValueError('the grading run belongs to other inputs')
    key = judge._read_jsonl(grading_inputs / 'key.jsonl')
    pairs = {row['pair']: row for row in judge._read_jsonl(grading_inputs / 'pairs.jsonl')}
    stored = json.loads((grading_inputs / 'universe.json').read_text())
    universe = {int(step): {query: list(docs) for query, docs in queries.items()} for step, queries in stored['snapshots'].items()}
    grades, answers = local_grades(grading_inputs, grading_output)
    ids, exact, primary, _, loaded = load(heldout_inputs, evidence, None)
    qrels, _ = load_test_qrels(Path(release) / 'files')
    official = qrels[judge.FAMILY]
    population = panel(official, ids)
    checked = validation(key, pairs, grades)
    effect = coverage(primary, exact, official, grades, PRIMARY_ARMS, population, universe)
    complete = require_complete(checked, effect, allow_partial=allow_partial)
    report = {'status': 'completed', 'kind': 'EXPLORATORY follow-up: judge validation and judgment coverage',
              'complete': complete, 'diagnostic_only': not complete,
              'pairs': len(pairs), 'answers': answers, 'thresholds': THRESHOLDS,
              'validation': checked, 'list_agreement': list_agreement(exact, official, grades, ids), 'coverage': effect,
              'scored_arms': sorted(PRIMARY_ARMS),
              'universe': {'pool_sets': stored['pool_sets'], 'sha256': contract['files']['universe.json'],
                           'occurrences': contract['universe_occurrences']},
              'binding': bound, 'inputs': loaded, 'grading': json.loads((grading_output / 'grading.json').read_text()),
              'source_sha256': checksum(__file__)}
    # A partial run decides nothing.
    report['compatible_with_the_official_assessor'] = bool(
        report['validation']['binary_agreement_threshold']['met'] and report['validation']['linear_weighted_kappa_threshold']['met']
        and report['list_agreement']['met']) if complete else None
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
    p = sub.add_parser('grade', help='grade every pair under one bound configuration')
    p.add_argument('inputs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--base-url', required=True)
    p.add_argument('--temperature', type=float, help='omit to send no sampling parameter, as the organisers did')
    p.add_argument('--server-config', type=Path, help='the model-server settings file, bound by checksum')
    p.add_argument('--workers', type=int, default=16)
    p = sub.add_parser('analyze', help='validation and coverage from a grading run')
    for name in ('grading_inputs', 'grading_output', 'heldout_inputs', 'release', 'evidence', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--allow-partial', action='store_true', help='write a diagnostic report although grades are missing')
    a = parser.parse_args()
    if a.command == 'prepare':
        r = prepare(a.release, a.ledger, a.pilot_inputs, a.output, pool_sets=tuple(a.pool_sets))
        r = {name: r[name] for name in ('pairs', 'validation_pairs', 'pool_pairs', 'pairs_in_both', 'pool_sets', 'universe_occurrences')}
    elif a.command == 'grade':
        params = {} if a.temperature is None else {'temperature': a.temperature}
        r = grade(a.inputs, a.output, base_url=a.base_url, params=params, server_config=a.server_config, workers=a.workers)
    else:
        r = analyze(a.grading_inputs, a.grading_output, a.heldout_inputs, a.release, a.evidence, a.output,
                    allow_partial=a.allow_partial)
        r = {'complete': r['complete'], 'compatible_with_the_official_assessor': r['compatible_with_the_official_assessor'],
             'validation': {k: r['validation'][k] for k in ('binary_agreement', 'linear_weighted_kappa')},
             'list_agreement': r['list_agreement']['met'], 'missing_grades': r['coverage']['missing_grades']}
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
