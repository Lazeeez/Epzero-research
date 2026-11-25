"""E4: the full-snapshot insertion sample on the BGE vectors (amendment 3 of EXTENSION_AMENDMENTS.md).

    python -m research.longeval.bge_sample BGE_VECTORS QWEN_SAMPLE_METADATA OUTPUT

Writes what `insertion_pilot prepare-full --draw 0` writes, in the same layout and by the same rules, with
the BGE document and query vectors in place of the released ones. `insertion_pilot.prepare_full` is frozen
and reads only the released Qwen archives, so this module repeats its steps:

- document order, arrival snapshot (`cohort.npy`) and query order come from the Qwen order rerun's saved
  sample metadata (`doc-ids.txt.gz`, `cohort.npy`, `query-ids.json`, `confirm-query-ids.json`); the BGE
  document IDs must equal that order and the BGE query ID sets must equal the Qwen ones;
- the lexical order key, and exact neighbours as float64 inner products by descending score, then
  descending document ID;
- in place of the check against the study's held-out references (none exist for BGE), the exact
  neighbours of a fixed subset of queries are recomputed by a full sort and must agree exactly.

BGE_VECTORS is the encoding job's `vectors` folder; its files must have the SHA-256 recorded in
`PINNED` (the report of job 7290861) unless `pinned=None`. Everything written is derived from LongEval
data and is never committed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

# The encoding report of job 7290861 (package longeval-bge-04), `files`.
PINNED = {
    'documents-snapshot-3.npy': '843758a80d53f91cdca310b1d293af88a92434e603f9b77bac44bd782163ca58',
    'documents-snapshot-3-ids.json': '128397e7820d24f07558693dbc66ff4ac4e7fb844372f37a37d0ec7b1e05e437',
    'queries-test.npy': 'f6324766c91ee5f9f2e25ce2ec8e93abd44f25ba3f1b6be540dd60be5862eb09',
    'queries-test-ids.json': 'a264febaf73bd0da5a5cfad10f36b7d91c22dd72360462ba3bf4caec9d2ee35b',
    'queries-training.npy': '44574a406bef1d131ce2bcffa25bac38a311603d097dbafbd7de2dea1707b4b5',
    'queries-training-ids.json': 'f6ead58e83b74e74737be2aad2c3d41c456d24adf03597702a7edbbfe00319b2',
}
CHECKED_QUERIES = 25


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def _ordered(ids, vectors, wanted, what):
    if sorted(ids) != sorted(wanted) or len(set(ids)) != len(ids):
        raise ValueError(f'BGE {what} query IDs differ from the Qwen sample\'s')
    position = {q: i for i, q in enumerate(ids)}
    return np.ascontiguousarray(vectors[[position[q] for q in wanted]], dtype=np.float32)


def prepare(bge, metadata, output, *, pinned=PINNED, checked=CHECKED_QUERIES):
    bge, metadata, output = Path(bge), Path(metadata), Path(output)
    if pinned is not None:
        for name, expected in pinned.items():
            if expected is None or sha256(bge / name) != expected:
                raise ValueError(f'{name} is not the verified BGE file')
    with gzip.open(metadata / 'doc-ids.txt.gz', 'rt', encoding='utf-8') as handle:
        ids = handle.read().split('\n')[:-1]
    if json.loads((bge / 'documents-snapshot-3-ids.json').read_text()) != ids:
        raise ValueError('BGE document order differs from the Qwen sample\'s')
    cohort = np.load(metadata / 'cohort.npy')
    query_ids = json.loads((metadata / 'query-ids.json').read_text())
    confirm_ids = json.loads((metadata / 'confirm-query-ids.json').read_text())
    test_vectors = _ordered(json.loads((bge / 'queries-test-ids.json').read_text()),
                            np.load(bge / 'queries-test.npy'), query_ids, 'test')
    confirm_vectors = _ordered(json.loads((bge / 'queries-training-ids.json').read_text()),
                               np.load(bge / 'queries-training.npy'), confirm_ids, 'training')
    documents = np.load(bge / 'documents-snapshot-3.npy', mmap_mode='r')
    n, dim = documents.shape
    if len(ids) != n or cohort.shape != (n,):
        raise ValueError('document count differs between the BGE vectors and the Qwen sample')
    output.mkdir(parents=True, exist_ok=False)
    # From here as insertion_pilot.prepare_full, draw 0, with the BGE vectors.
    q64 = np.vstack([test_vectors, confirm_vectors]).astype(np.float64)
    rows = np.arange(n)
    vectors = np.lib.format.open_memmap(output / 'vectors.npy', mode='w+', dtype=np.float32, shape=(n, dim))
    scores = np.empty((len(q64), n), dtype=np.float64)
    for start in range(0, n, 20_000):
        block = np.asarray(documents[rows[start:start + 20_000]], dtype=np.float32)
        vectors[start:start + len(block)] = block
        scores[:, start:start + len(block)] = (block.astype(np.float64) @ q64.T).T
    vectors.flush()
    del vectors
    lexical = np.empty(n, dtype=np.int64)
    lexical[np.argsort(np.array(ids, dtype=object), kind='stable')] = np.arange(n)
    top = np.empty((len(q64), 100), dtype=np.int32)
    for qi in range(len(q64)):
        row = scores[qi]
        candidates = np.argpartition(-row, 255)[:256]
        candidates = candidates[np.lexsort((-lexical[candidates], -row[candidates]))]
        top[qi] = candidates[:100]
    # In place of the study's reference check: a full sort of every document for a fixed subset of queries.
    subset = np.unique(np.linspace(0, len(q64) - 1, min(checked, len(q64))).round().astype(int))
    agree = [bool(np.array_equal(np.lexsort((-lexical, -scores[qi]))[:100], top[qi])) for qi in subset]
    del scores
    nt = len(query_ids)
    np.save(output / 'cohort.npy', cohort)
    np.save(output / 'lexical-rank.npy', lexical)
    np.save(output / 'queries.npy', test_vectors)
    np.save(output / 'exact-top100.npy', top[:nt])
    (output / 'query-ids.json').write_text(json.dumps(query_ids) + '\n')
    with gzip.open(output / 'doc-ids.txt.gz', 'wt', encoding='utf-8') as handle:
        handle.write('\n'.join(ids) + '\n')
    np.save(output / 'confirm-queries.npy', confirm_vectors)
    np.save(output / 'confirm-exact-top100.npy', top[nt:])
    (output / 'confirm-query-ids.json').write_text(json.dumps(confirm_ids) + '\n')
    summary = {'count': n, 'cohort_sizes': {str(c): int((cohort == c).sum()) for c in (1, 2, 3)},
               'queries': nt, 'metric': 'ip', 'draw': 0, 'collection': 'snapshot 3', 'documents_in_snapshot': n,
               'ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(), 'confirm_queries': len(confirm_ids),
               'encoder': 'BAAI/bge-base-en-v1.5', 'dimension': int(dim),
               'bge_files_sha256': {name: sha256(bge / name) for name in PINNED} if pinned is not None else None,
               'exact_full_sort_check': {'queries': len(subset), 'identical': int(sum(agree)), 'passed': all(agree)}}
    (output / 'sample.json').write_text(json.dumps(summary, indent=2) + '\n')
    if not all(agree):
        raise ValueError('exact neighbours differ from a full sort')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('bge', type=Path)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('output', type=Path)
    a = parser.parse_args()
    summary = prepare(a.bge, a.metadata, a.output)
    print(json.dumps({k: summary[k] for k in ('count', 'cohort_sizes', 'queries', 'confirm_queries', 'exact_full_sort_check')}))


if __name__ == '__main__':
    main()
