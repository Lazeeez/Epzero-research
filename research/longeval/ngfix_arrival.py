"""Screen A2b: NGFix against the arrival bias (docs/screen-a2b-ngfix.md; design recorded before this code).

    python -m research.longeval.ngfix_arrival prepare SAMPLE OUT --order appended|shuffled
    ngfix-driver OUT/base.fbin OUT/queries.fbin OUT/calibration_top200.ibin RUN        (the unchanged adapter)
    python -m research.longeval.ngfix_arrival score OUT RUN SAMPLE SCORED.json

`prepare` writes the adapter's inputs for one sample and order: the sample's vectors in insertion order, its queries
split by seed 2029 into a training half (first in the query file, the only queries NGFix and RFix see) and a test
half, and the training queries' exact top 200 over the ordered vectors (float64 inner product, ties by position).
`score` maps the adapter's rankings back to sample positions and scores the test queries against the sample's
exact top 100 by arrival cohort, for every variant, with paired query-bootstrap differences against `base`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .accumulation import miss_table
from .insertion_pilot import insertion_order
from .insertion_strata import mantel_haenszel, risk_difference

SPLIT_SEED, ORDER_SEED = 2029, 11
VARIANTS = ('base', 'ngfix', 'rfix', 'ngfix_rfix')
EFS = (100, 200, 400, 800)
DRAWS, RNG = 5000, 20261129


def _write_matrix(path, rows, dtype, chunk=20_000):
    with open(path, 'wb') as f:
        np.array([len(rows), rows.shape[1]], dtype='<u4').tofile(f)
        for start in range(0, len(rows), chunk):
            np.ascontiguousarray(rows[start:start + chunk], dtype=dtype).tofile(f)


def split(count, seed=SPLIT_SEED):
    """Training and test query indices: a seeded permutation, the first floor(count / 2) for training."""
    perm = np.random.default_rng(seed).permutation(count)
    half = count // 2
    return np.sort(perm[:half]), np.sort(perm[half:])


def exact_top(base, queries, k=200, chunk=50_000):
    """Exact inner-product top k over `base` (rows in insertion order): float32 scan for 2k candidates, re-scored in
    float64, best first, ties by ascending position, so that the adapter's double-precision order check holds."""
    q32 = np.ascontiguousarray(queries, dtype=np.float32)
    best_ids = np.empty((len(queries), 0), dtype=np.int64)
    best_key = np.empty((len(queries), 0), dtype=np.float32)
    for start in range(0, len(base), chunk):
        block = np.asarray(base[start:start + chunk], dtype=np.float32)
        key = np.concatenate([best_key, -(q32 @ block.T)], axis=1)
        ids = np.concatenate([best_ids, np.broadcast_to(np.arange(start, start + len(block)), (len(queries), len(block)))], axis=1)
        keep = np.argpartition(key, min(2 * k, key.shape[1] - 1), axis=1)[:, :2 * k]
        best_key, best_ids = np.take_along_axis(key, keep, axis=1), np.take_along_axis(ids, keep, axis=1)
    top = np.empty((len(queries), k), dtype=np.int64)
    for qi in range(len(queries)):
        ids = np.sort(best_ids[qi])
        score = np.asarray(base[ids], dtype=np.float64) @ np.asarray(queries[qi], dtype=np.float64)
        top[qi] = ids[np.lexsort((ids, -score))[:k]]
    return top


def prepare(sample, output, *, order):
    sample, output = Path(sample), Path(output)
    meta = json.loads((sample / 'sample.json').read_text())
    if meta.get('metric', 'ip') != 'ip':
        raise ValueError('the adapter is inner product only')
    vectors = np.load(sample / 'vectors.npy', mmap_mode='r')
    queries = np.load(sample / 'queries.npy')
    sequence = insertion_order(np.load(sample / 'lexical-rank.npy'), np.load(sample / 'cohort.npy'), order, ORDER_SEED)
    train, test = split(len(queries))
    output.mkdir(parents=True, exist_ok=False)
    ordered = np.asarray(vectors[sequence], dtype=np.float32)
    _write_matrix(output / 'base.fbin', ordered, '<f4')
    _write_matrix(output / 'queries.fbin', np.concatenate([queries[train], queries[test]]), '<f4')
    _write_matrix(output / 'calibration_top200.ibin', exact_top(ordered, queries[train]), '<i4')
    np.save(output / 'sequence.npy', sequence)
    np.save(output / 'train.npy', train)
    np.save(output / 'test.npy', test)
    info = {'design': 'docs/screen-a2b-ngfix.md', 'sample': sample.name, 'order': order, 'order_seed': ORDER_SEED,
            'split_seed': SPLIT_SEED, 'count': int(len(sequence)), 'dimension': int(vectors.shape[1]),
            'train_queries': int(len(train)), 'test_queries': int(len(test))}
    (output / 'prepared.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


def read_rankings(path, n_queries, efs=EFS):
    """{(variant, ef): [query, 100] adapter IDs} from the adapter's rankings.tsv."""
    data = np.genfromtxt(path, delimiter='\t', names=True, dtype=None, encoding='utf-8')
    out = {}
    for variant in VARIANTS:
        for ef in efs:
            rows = data[(data['variant'] == variant) & (data['ef'] == ef)]
            table = np.full((n_queries, 100), -1, dtype=np.int64)
            table[rows['query'], rows['rank'] - 1] = rows['id']
            if (table < 0).any():
                raise ValueError(f'incomplete rankings for {variant} at ef {ef}')
            out[(variant, ef)] = table
    return out


def score(prepared, run, sample, *, ef=200):
    prepared, run, sample = Path(prepared), Path(run), Path(sample)
    info = json.loads((prepared / 'prepared.json').read_text())
    sequence, test = np.load(prepared / 'sequence.npy'), np.load(prepared / 'test.npy')
    n_train = info['train_queries']
    cohort = np.load(sample / 'cohort.npy').astype(np.int64) - 1
    exact = np.load(sample / 'exact-top100.npy')[test]
    rankings = read_rankings(run / 'rankings.tsv', n_train + len(test))
    draws = np.random.default_rng(RNG).integers(0, len(test), (DRAWS, len(test)))
    tables, out = {}, {'kind': 'EXPLORATORY screen A2b (docs/screen-a2b-ngfix.md), aggregates only', 'prepared': info,
                       'ef': ef, 'variants': {}, 'paired_against_base': {}}
    for variant in VARIANTS:
        found = sequence[rankings[(variant, ef)][n_train:]]          # adapter ID -> sample position
        tables[variant] = table = miss_table(found, exact, cohort, 3)
        recall = {str(e): float(np.mean([np.isin(exact[q], sequence[rankings[(variant, e)][n_train + q]]).mean()
                                         for q in range(len(test))])) for e in EFS}
        out['variants'][variant] = {
            'within_query_ratio': float(mantel_haenszel(table, 3, 1)),
            'ratio_interval_95': np.quantile([mantel_haenszel(table[r], 3, 1) for r in draws], [.025, .975]).tolist(),
            'newest_excess_pp': float(100 * risk_difference(table, 3, 1)),
            'excess_interval_95_pp': np.quantile([100 * risk_difference(table[r], 3, 1) for r in draws], [.025, .975]).tolist(),
            'recall_at_100': recall}
    for variant in VARIANTS[1:]:
        diff = [100 * (risk_difference(tables[variant][r], 3, 1) - risk_difference(tables['base'][r], 3, 1)) for r in draws]
        out['paired_against_base'][variant] = {
            'excess_change_pp': out['variants'][variant]['newest_excess_pp'] - out['variants']['base']['newest_excess_pp'],
            'interval_95_pp': np.quantile(diff, [.025, .975]).tolist(),
            'recall_change_pp_at_ef': 100 * (out['variants'][variant]['recall_at_100'][str(ef)] - out['variants']['base']['recall_at_100'][str(ef)])}
    stages = (run / 'stages.tsv').read_text().splitlines()[1:]
    out['stages'] = [dict(zip(('variant', 'stage', 'seconds', 'serialized_bytes'), line.split('\t'))) for line in stages]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('sample', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--order', choices=('appended', 'shuffled'), required=True)
    s = sub.add_parser('score')
    for name in ('prepared', 'run', 'sample', 'output'):
        s.add_argument(name, type=Path)
    a = parser.parse_args()
    if a.command == 'prepare':
        print(json.dumps(prepare(a.sample, a.output, order=a.order), indent=2))
    else:
        out = score(a.prepared, a.run, a.sample)
        a.output.write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps({v: {k: out['variants'][v][k] for k in ('within_query_ratio', 'newest_excess_pp')} for v in VARIANTS}, indent=2))


if __name__ == '__main__':
    main()
