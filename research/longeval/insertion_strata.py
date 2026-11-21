"""Miss ratios of later arrivals within the same query (EXPLORATORY).

Written on 21 November 2026, after the full-scale insertion-order results were seen. It is not
part of the design recorded in docs/protocol/INSERTION_ORDER_DESIGN_RECORD.md.

The rank-standardized ratio of `insertion_pilot` pools all queries: it compares the misses of a
later arrival group with the misses expected at the snapshot-1 group's rate in each exact-rank
band. A query whose neighbours are mostly later documents can be a harder query in itself, and
the pooled ratio then mixes the query with the documents. This module stratifies by query and
rank band, so a later document is only compared with snapshot-1 documents that are neighbours of
the same query at the same depth:

- the Mantel-Haenszel risk ratio, which stays consistent when strata are many and small;
- the Mantel-Haenszel risk difference, as a second estimator of the same contrast.

It reads the per-query count tables that `insertion_pilot run` writes ([query, cohort, band,
(neighbours, missed)]) and, for the held-out arms, counts the same table from the returned runs.
Outputs hold counts and ratios only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .insertion_pilot import BANDS, _draws, _ratio

ORDERS = ('appended', 'shuffled', 'reversed')
TARGETS = (('cohort3_vs_cohort1', 3), ('cohort2_vs_cohort1', 2), ('later_vs_cohort1', (2, 3)))


def _cells(table, target, reference):
    """Per stratum [query, band]: later neighbours and misses, reference neighbours and misses."""
    target = [c - 1 for c in ((target,) if isinstance(target, int) else target)]
    return (table[..., 0][:, target].sum(axis=1), table[..., 1][:, target].sum(axis=1),
            table[:, reference - 1, :, 0], table[:, reference - 1, :, 1])


def mantel_haenszel(table, target, reference=1):
    """Risk ratio of the target cohorts against the reference cohort, stratified by query and rank band."""
    n1, a, n0, c = _cells(table, target, reference)
    total = np.maximum(n1 + n0, 1)
    below = (c * n1 / total).sum()
    return float((a * n0 / total).sum() / below) if below else float('nan')


def risk_difference(table, target, reference=1):
    """Mantel-Haenszel difference of miss rates (target minus reference) over the same strata."""
    n1, a, n0, c = _cells(table, target, reference)
    total = np.maximum(n1 + n0, 1)
    weight = (n1 * n0 / total).sum()
    return float(((a * n0 - c * n1) / total).sum() / weight) if weight else float('nan')


def tables(folder, ef=None):
    """{(order, seed): per-query table} for the builds of one implementation."""
    out = {}
    for path in sorted(Path(folder).glob('*/report.json')):
        report = json.loads(path.read_text())
        named = path.parent / f'per-query-ef{ef}.npy'
        out[(report['order'], report['seed'])] = np.load(named if ef is not None and named.exists() else path.parent / 'per-query.npy')
    if not out:
        raise ValueError(f'no build in {folder}')
    return out


def _item(value, sample):
    return {'value': float(value), 'interval_95': np.quantile(sample, [.025, .975]).tolist()}


def _seed_mean(builds, order, statistic, rows):
    mine = [table for (o, _), table in sorted(builds.items()) if o == order]
    point = np.mean([statistic(table) for table in mine])
    boot = np.mean([[statistic(table[r]) for r in rows] for table in mine], axis=0)
    return point, boot


def orders(folder, *, ef=None, draws=5000, seed=20261120):
    """Pooled and within-query ratios of each insertion order, as the mean over seeds with a paired query bootstrap."""
    builds = tables(folder, ef)
    rows = _draws(len(next(iter(builds.values()))), draws, seed)
    out = {}
    for name, target in TARGETS:
        block = {}
        for label, statistic in (('pooled_ratio', lambda t: _ratio(t.sum(axis=0), target, 1)),
                                 ('within_query_ratio', lambda t: mantel_haenszel(t, target)),
                                 ('within_query_rate_difference', lambda t: risk_difference(t, target))):
            found = {order: _seed_mean(builds, order, statistic, rows) for order in ORDERS if any(o == order for o, _ in builds)}
            block[label] = {order: _item(*pair) for order, pair in found.items()}
            if 'appended' in found and 'reversed' in found:
                block[label]['as_arrived_minus_reversed'] = _item(found['appended'][0] - found['reversed'][0],
                                                                  found['appended'][1] - found['reversed'][1])
        out[name] = block
    return out


def difficulty(folder, *, order='shuffled', ef=None):
    """Are the queries with many later neighbours harder in themselves?

    Queries are split into thirds by the share of their exact neighbours that arrived after
    snapshot 1; the miss rate of each arrival group is given for each third, in the builds of one
    order (shuffled by default, where position is independent of the document).
    """
    total = sum(table for (o, _), table in tables(folder, ef).items() if o == order)
    share = total[:, 1:, :, 0].sum(axis=(1, 2)) / total[..., 0].sum(axis=(1, 2))
    reference = total[:, 0, :, 1].sum(axis=1) / np.maximum(total[:, 0, :, 0].sum(axis=1), 1)
    thirds = []
    for part in np.array_split(np.argsort(share, kind='stable'), 3):
        counts = total[part].sum(axis=0)
        thirds.append({'queries': int(len(part)), 'mean_share_of_later_neighbours': float(share[part].mean()),
                       'miss_rate_by_arrival_group': [float(counts[c, :, 1].sum() / counts[c, :, 0].sum()) if counts[c, :, 0].sum() else None
                                                      for c in range(3)]})
    return {'order': order, 'thirds_by_share_of_later_neighbours': thirds,
            'correlation_share_with_snapshot1_miss_rate': float(np.corrcoef(share, reference)[0, 1])}


def between(first, second, *, ef=None, draws=5000, seed=20261120):
    """Ratios in `first` minus the mean of the collections in `second`, paired by query (same queries, same draws)."""
    first, second = tables(first, ef), [tables(folder, ef) for folder in second]
    rows = _draws(len(next(iter(first.values()))), draws, seed)
    out = {}
    for label, statistic in (('pooled_ratio', lambda t: _ratio(t.sum(axis=0), 3, 1)), ('within_query_ratio', lambda t: mantel_haenszel(t, 3))):
        out[label] = {}
        for order in ORDERS:
            point, boot = _seed_mean(first, order, statistic, rows)
            others = [_seed_mean(builds, order, statistic, rows) for builds in second]
            other_point, other_boot = np.mean([p for p, _ in others]), np.mean([b for _, b in others], axis=0)
            out[label][order] = {'first': _item(point, boot), 'second_mean': _item(other_point, other_boot),
                                 'difference': _item(point - other_point, boot - other_boot)}
    return out


def heldout(inputs, evidence, dense, *, step=3, draws=5000, seed=20261120):
    """The same two ratios for the held-out arms at one snapshot: later arrivals against snapshot-1 arrivals."""
    from .heldout import ROLE, load_heldout
    from .heldout_analysis import SEEDS, collect
    from .heldout_exploratory import cohorts
    queries, refs, _, _ = load_heldout(Path(inputs))
    ids = queries[ROLE].ids
    exact = {s: {q: refs[s - 1][q] for q in ids} for s in (1, 2, 3)}
    _, runs, _ = collect(sorted(Path(evidence).rglob('report.json')), ROLE, ids, exact)
    first = cohorts(dense)
    arms = {'epzero_hnsw_ef200': ('epzero', 'incremental', 'frozen'), 'faiss_hnsw_ef200': ('faiss_hnsw', 'incremental', 'frozen'),
            'ivf_frozen_training': ('faiss_ivf', 'frozen_incremental', 'current_calibration'),
            'ivf_refreshed_training': ('faiss_ivf', 'refreshed_rebuild', 'current_calibration')}
    rows = _draws(len(ids), draws, seed)
    out = {}
    for name, (backend, mode, regime) in arms.items():
        builds = {}
        for run_seed in SEEDS:
            run = runs[(backend, mode, run_seed, step, regime)]
            table = np.zeros((len(ids), 3, len(BANDS), 2), dtype=np.int64)
            for qi, q in enumerate(ids):
                returned = {d for d, _ in run[q]}
                for rank, (doc, _) in enumerate(exact[step][q], 1):
                    band = next(i for i, b in enumerate(BANDS) if b[0] <= rank <= b[1])
                    table[qi, first[doc] - 1, band, 0] += 1
                    table[qi, first[doc] - 1, band, 1] += doc not in returned
            builds[('appended', run_seed)] = table
        out[name] = {label: _item(*_seed_mean(builds, 'appended', statistic, rows))
                     for label, statistic in (('pooled_ratio', lambda t: _ratio(t.sum(axis=0), (2, 3), 1)),
                                              ('within_query_ratio', lambda t: mantel_haenszel(t, (2, 3))),
                                              ('within_query_rate_difference', lambda t: risk_difference(t, (2, 3))))}
    return {'snapshot': step, 'queries': len(ids), 'later_group': 'snapshots 2 and 3 together', 'arms': out}


def analyze(collections, output, *, ef=200, compare=None):
    """One aggregate file: {collection: {implementation: orders, difficulty}}, and the comparison across sizes."""
    out = {'kind': 'EXPLORATORY within-query miss ratios by insertion order (aggregates only)', 'ef': ef,
           'strata': 'query x exact-rank band', 'bands': [f'{lo}-{hi}' for lo, hi in BANDS],
           'bootstrap': {'draws': 5000, 'seed': 20261120, 'unit': 'query'}, 'collections': {}}
    for name, root in collections.items():
        sample = json.loads((Path(root) / 'sample' / 'sample.json').read_text()) if (Path(root) / 'sample' / 'sample.json').exists() else None
        block = {'documents': sample and sample['count'], 'implementations': {}}
        for folder in sorted(p for p in Path(root).iterdir() if p.is_dir() and any(p.glob('*/report.json'))):
            block['implementations'][folder.name] = {'ratios': orders(folder, ef=ef), 'query_difficulty': difficulty(folder, ef=ef)}
        out['collections'][name] = block
    if compare:
        first, second = compare
        out['between_collections'] = {
            'first': first, 'second': list(second),
            'implementations': {backend: between(Path(collections[first]) / backend, [Path(collections[name]) / backend for name in second], ef=ef)
                                for backend in out['collections'][first]['implementations']}}
    Path(output).write_text(json.dumps(out, indent=2) + '\n')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('analyze', help='pooled and within-query ratios of collected builds')
    p.add_argument('output', type=Path)
    p.add_argument('collections', nargs='+', help='NAME=FOLDER, a folder holding one sub-folder of builds per implementation')
    p.add_argument('--ef', type=int, default=200)
    p.add_argument('--compare', nargs='+', help='FIRST SECOND [SECOND ...]: FIRST minus the mean of the others, paired by query')
    p = sub.add_parser('heldout', help='the held-out arms at one snapshot')
    for name in ('inputs', 'evidence', 'dense', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--snapshot', type=int, default=3)
    a = parser.parse_args()
    if a.command == 'analyze':
        collections = dict(item.split('=', 1) for item in a.collections)
        r = analyze(collections, a.output, ef=a.ef, compare=(a.compare[0], a.compare[1:]) if a.compare else None)
        print(json.dumps({name: list(block['implementations']) for name, block in r['collections'].items()}, indent=2))
    else:
        r = heldout(a.inputs, a.evidence, a.dense, step=a.snapshot)
        r['kind'] = 'EXPLORATORY within-query miss ratios of the held-out arms (aggregates only)'
        a.output.write_text(json.dumps(r, indent=2) + '\n')
        print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
