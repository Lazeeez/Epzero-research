"""The judging pool's two grading folders (amendment 5 of EXTENSION_AMENDMENTS.md).

    python -m research.longeval.grading_inputs primary RELEASE LEDGER PILOT_INPUTS OUTPUT
    python -m research.longeval.grading_inputs substitutions RELEASE LEDGER ORDER_BUILDS ORDER_METADATA OUTPUT

`primary` is the frozen `coverage.prepare` with the pooled sets of decision 3 (`reported`, `+all_lists`,
`+extension`). `substitutions` writes, in the same input format, the order rerun's top-10 substitutions at
ef 200 that the primary pool does not already hold: documents in a build's top 10 that are not in exact
search's top 10, over every build under ORDER_BUILDS (`<backend>/<order>-<seed>/labels-ef200.npy`), mapped
to documents by ORDER_METADATA (`doc-ids.txt.gz`, `query-ids.json`, `exact-top100.npy`). Their grades serve
the arrival-order analyses; they enter no primary estimate. Everything written is derived from LongEval
data and is never committed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from . import coverage, judge
from .release import checksum, read_queries
from .storage import atomic_json

POOL_SETS = ('reported', '+all_lists', '+extension')        # decision 3
WIDTH, TOP = 200, 10
SET_NAME = 'e2b_substitutions'


def primary(release, ledger, pilot_inputs, output):
    return coverage.prepare(release, ledger, pilot_inputs, output, pool_sets=POOL_SETS)


def pooled(ledger, pool_sets=POOL_SETS):
    with Path(ledger).open(newline='') as handle:
        return {(row['query'], row['document']) for row in csv.DictReader(handle) if row['pooled_set'] in pool_sets}


def substitutions(builds, metadata):
    """{(query, document)} in some build's top 10 at ef 200 and not in exact search's top 10; and the builds read."""
    metadata = Path(metadata)
    with gzip.open(metadata / 'doc-ids.txt.gz', 'rt', encoding='utf-8') as handle:
        docs = handle.read().split('\n')[:-1]
    queries = json.loads((metadata / 'query-ids.json').read_text())
    exact = np.load(metadata / 'exact-top100.npy')[:, :TOP]
    found, read = set(), []
    for path in sorted(Path(builds).glob(f'*/*-*/labels-ef{WIDTH}.npy')):
        top = np.load(path)[:, :TOP]
        if top.shape[0] != len(queries):
            raise ValueError(f'{path}: another query population')
        for row, query in enumerate(queries):
            for position in top[row]:
                if position >= 0 and position not in exact[row]:
                    found.add((query, docs[position]))
        read.append(f'{path.parent.parent.name}/{path.parent.name}')
    if not read:
        raise ValueError('no order-rerun build found')
    return found, read


def write_substitutions(release, ledger, builds, metadata, output, *, seed=coverage.SEED):
    release, output = Path(release), Path(output)
    found, read = substitutions(builds, metadata)
    already = pooled(ledger)
    pairs = sorted(found - already)
    if not pairs:
        raise ValueError('every substitution is already in the primary pool')
    queries = read_queries(release / 'files' / judge.TEST_FILE)
    if checksum(release / 'files' / judge.TEST_FILE) != judge.TEST_SHA256:
        raise ValueError('test query release identity mismatch')
    from .heldout_analysis import load_test_qrels
    qrels, _ = load_test_qrels(release / 'files')
    official = judge.official_pairs(qrels[judge.FAMILY])
    order = list(pairs)
    random.Random(seed).shuffle(order)                       # as coverage.prepare orders its pairs
    texts = judge.document_texts(release / 'documents', {doc for _, doc in order})
    output.mkdir(parents=True, exist_ok=False)
    rows = [{'pair': n, 'query_id': q, 'doc_id': d, 'query': queries[q], 'text': texts[d]} for n, (q, d) in enumerate(order, 1)]
    key = [{'pair': n, 'validation': False, 'pool': True,
            'official_grade': official[(q, d)][0] if (q, d) in official else None,
            'selection_probability': None, 'stratum_pairs': None, 'first_graded_snapshot': None}
           for n, (q, d) in enumerate(order, 1)]
    judge._write_jsonl(output / 'pairs.jsonl', rows)
    judge._write_jsonl(output / 'key.jsonl', key)
    universe = {'1': {}, '2': {}, '3': {}}
    for query, doc in pairs:
        universe['3'].setdefault(query, []).append(doc)
    atomic_json(output / 'universe.json', {'pool_sets': [SET_NAME], 'depth': coverage.DEPTH, 'snapshots': universe})
    contract = {'status': 'prepared', 'purpose': 'the order rerun\'s top-10 substitutions outside the primary pool (amendment 5)',
                'pairs': len(rows), 'validation_pairs': 0, 'pool_pairs': len(rows), 'pairs_in_both': 0,
                'pool_sets': [SET_NAME], 'seed': seed, 'substitutions': len(found), 'already_in_primary_pool': len(found & already),
                'builds': read, 'width': WIDTH, 'depth': TOP,
                'prompt_sha256': hashlib.sha256(judge.prompt('{query}', '{document}').encode()).hexdigest(),
                'ledger_sha256': checksum(ledger), 'source_sha256': checksum(__file__),
                'files': {name: checksum(output / name) for name in coverage.INPUT_FILES}}
    atomic_json(output / 'inputs.json', contract)
    coverage.read_inputs(output)                              # the frozen grading accepts it
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('primary')
    for name in ('release', 'ledger', 'pilot_inputs', 'output'):
        p.add_argument(name, type=Path)
    p = sub.add_parser('substitutions')
    for name in ('release', 'ledger', 'builds', 'metadata', 'output'):
        p.add_argument(name, type=Path)
    a = parser.parse_args()
    if a.command == 'primary':
        r = primary(a.release, a.ledger, a.pilot_inputs, a.output)
        r = {k: r[k] for k in ('pairs', 'validation_pairs', 'pool_pairs', 'pairs_in_both', 'pool_sets', 'universe_occurrences')}
    else:
        r = write_substitutions(a.release, a.ledger, a.builds, a.metadata, a.output)
        r = {k: r[k] for k in ('pairs', 'substitutions', 'already_in_primary_pool', 'builds')}
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
