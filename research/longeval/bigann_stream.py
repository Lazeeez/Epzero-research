"""A public streaming workload in the insertion study's format (design: scripts/pace/BIGANN_STREAM.md).

    python -m research.longeval.bigann_stream prepare DATA.fbin QUERIES.fbin GT RUNBOOK.yaml OUTPUT
    python -m research.longeval.bigann_stream analyze SAMPLE epzero=FOLDER faiss_hnsw=FOLDER [--engine FOLDER] --output OUT.json

`prepare` turns Big-ANN's clustered MSTuring-10M stream into the sample format that
`insertion_pilot run` and `engine_arm` read: vectors in file order, an arrival cohort per vector (the
third of the stream in which its cluster's block starts), the order within cohorts (file order), the
queries and their exact top-100 (computed here, checked against the track's own file). It also writes
each vector's block and its position within the block. The runbook is read as data with a small
parser; no code from the benchmark's repository is run.

`analyze` computes the measures fixed in the design from the builds' returned IDs and count tables:
the paired cohort-gap change between orders (or between engine arms), the within-block position
difference, and the global position difference in shuffled builds. Aggregates only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .insertion_pilot import BANDS
from .insertion_strata import risk_difference
from . import engine_arm_analysis, order_pairs

DATASET = 'msturing-10M-clustered'
TENTHS = 10
GT_MEAN_OVERLAP, GT_MIN_OVERLAP = 0.999, 0.98


def sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_bin(path):
    """Big-ANN .fbin: uint32 rows, uint32 dimension, then row-major float32; memory-mapped."""
    rows, dim = (int(v) for v in np.fromfile(path, dtype='<u4', count=2))
    data = np.memmap(path, dtype='<f4', mode='r', offset=8)
    if rows <= 0 or dim <= 0 or data.size != rows * dim:
        raise ValueError(f'{path}: header ({rows} x {dim}) does not match the file size')
    return data.reshape(rows, dim)


def read_gt(path):
    """Big-ANN ground truth: uint32 queries, uint32 k, then the IDs (uint32) and distances (float32)."""
    queries, k = (int(v) for v in np.fromfile(path, dtype='<u4', count=2))
    raw = np.fromfile(path, dtype='<u4', offset=8)
    if raw.size != 2 * queries * k:
        raise ValueError(f'{path}: header ({queries} x {k}) does not match the file size')
    return raw[:queries * k].reshape(queries, k).astype(np.int64)


def parse_runbook(text, dataset=DATASET):
    """The insert ranges of one dataset's runbook, in step order: [(start, end), ...], plus max_pts.

    Accepts the runbook's simple layout: a dataset key at column 0, then `max_pts:` and numbered steps,
    each with `operation:` and, for inserts and deletes, `start:` and `end:`."""
    steps, current, inside, max_pts = {}, None, False, None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if not line[0].isspace():
            inside = line.split(':', 1)[0].strip().strip('"\'') == dataset
            current = None
            continue
        if not inside:
            continue
        if m := re.fullmatch(r'\s+(\d+)\s*:\s*', line):
            current = steps.setdefault(int(m.group(1)), {})
        elif m := re.fullmatch(r'\s+max_pts\s*:\s*(\d+)\s*', line):
            max_pts = int(m.group(1))
        elif current is not None and (m := re.fullmatch(r'\s+(operation|start|end)\s*:\s*["\']?([\w]+)["\']?\s*', line)):
            current[m.group(1)] = m.group(2)
    if not steps:
        raise ValueError(f'no steps for {dataset} in the runbook')
    operations = [steps[k] for k in sorted(steps)]
    if any(s.get('operation') not in ('insert', 'search') for s in operations):
        raise ValueError('the runbook has operations other than insert and search')
    blocks = [(int(s['start']), int(s['end'])) for s in operations if s['operation'] == 'insert']
    return blocks, max_pts


def check_blocks(blocks, rows):
    if not blocks or blocks[0][0] != 0 or blocks[-1][1] != rows:
        raise ValueError('the insert ranges do not cover the data')
    for (s0, e0), (s1, e1) in zip(blocks, blocks[1:] + [(blocks[-1][1], None)]):
        if e0 <= s0 or e0 != s1:
            raise ValueError('the insert ranges are not contiguous')


def cohorts(blocks, rows):
    """Cohort per vector (1-3, by the third in which its block starts), block index, tenth within block."""
    first, second = (rows + 2) // 3, (2 * rows + 2) // 3
    cohort = np.empty(rows, dtype=np.int8)
    block = np.empty(rows, dtype=np.int16)
    tenth = np.empty(rows, dtype=np.int8)
    for b, (start, end) in enumerate(blocks):
        cohort[start:end] = 1 if start < first else 2 if start < second else 3
        block[start:end] = b
        tenth[start:end] = (np.arange(end - start) * TENTHS // (end - start)).astype(np.int8)
    return cohort, block, tenth


def exact_top100(data, queries, *, candidates=200, chunk=1_000_000):
    """Exact Euclidean top-100: float32 search for `candidates`, re-scored in float64, ties by position."""
    import faiss
    index = faiss.IndexFlatL2(data.shape[1])
    for start in range(0, len(data), chunk):
        index.add(np.ascontiguousarray(data[start:start + chunk], dtype=np.float32))
    _, found = index.search(np.ascontiguousarray(queries, dtype=np.float32), candidates)
    if (found < 0).any():
        raise ValueError('fewer vectors than candidates')
    top = np.empty((len(queries), 100), dtype=np.int64)
    q64 = queries.astype(np.float64)
    for qi in range(len(queries)):
        ids = np.sort(found[qi])
        rows = np.asarray(data[ids], dtype=np.float64)
        distance = ((rows - q64[qi]) ** 2).sum(axis=1)
        top[qi] = ids[np.lexsort((ids, distance))[:100]]
    return top


def overlap(a, b):
    per = np.array([len(set(x.tolist()) & set(y.tolist())) / a.shape[1] for x, y in zip(a, b)])
    return {'mean': float(per.mean()), 'min': float(per.min()), 'identical_sets': int((per == 1).sum())}


def prepare(data_path, queries_path, gt_path, runbook_path, output):
    data_path, queries_path, gt_path, runbook_path, output = map(Path, (data_path, queries_path, gt_path, runbook_path, output))
    data, queries = read_bin(data_path), read_bin(queries_path)
    if data.shape[1] != queries.shape[1]:
        raise ValueError('data and queries differ in dimension')
    blocks, max_pts = parse_runbook(runbook_path.read_text())
    check_blocks(blocks, len(data))
    if max_pts is not None and max_pts != len(data):
        raise ValueError('the runbook is for another number of vectors')
    cohort, block, tenth = cohorts(blocks, len(data))
    exact = exact_top100(data, queries)
    reference = read_gt(gt_path)
    if reference.shape[0] != len(queries) or reference.shape[1] < 100:
        raise ValueError('the track ground truth does not cover the queries at depth 100')
    check = overlap(exact, reference[:, :100])
    check |= {'mean_required': GT_MEAN_OVERLAP, 'min_required': GT_MIN_OVERLAP,
              'passed': check['mean'] >= GT_MEAN_OVERLAP and check['min'] >= GT_MIN_OVERLAP}
    if not check['passed']:
        raise ValueError(f'exact targets disagree with the track ground truth: {check}')
    output.mkdir(parents=True, exist_ok=False)
    with open(output / 'vectors.npy', 'wb') as f:          # streamed: the file is several GB
        np.lib.format.write_array_header_1_0(f, {'descr': '<f4', 'fortran_order': False, 'shape': data.shape})
        for start in range(0, len(data), 1_000_000):
            np.ascontiguousarray(data[start:start + 1_000_000], dtype='<f4').tofile(f)
    np.save(output / 'cohort.npy', cohort)
    np.save(output / 'block.npy', block)
    np.save(output / 'block-tenth.npy', tenth)
    np.save(output / 'lexical-rank.npy', np.arange(len(data), dtype=np.int64))
    np.save(output / 'queries.npy', np.ascontiguousarray(queries, dtype=np.float32))
    np.save(output / 'exact-top100.npy', exact.astype(np.int32))
    summary = {'dataset': DATASET, 'count': int(len(data)), 'dimension': int(data.shape[1]), 'metric': 'l2',
               'queries': int(len(queries)), 'blocks': [[int(s), int(e)] for s, e in blocks],
               'cohort_sizes': {str(c): int((cohort == c).sum()) for c in (1, 2, 3)},
               'cohort_blocks': {str(c): sorted({int(b) for b, (s, _) in enumerate(blocks) if cohort[s] == c}) for c in (1, 2, 3)},
               'ground_truth_check': check,
               'files': {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size}
                         for p in (data_path, queries_path, gt_path, runbook_path)},
               'derived_sha256': {name: hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
                                  for name, arr in (('cohort', cohort), ('exact_top100', exact.astype(np.int32)))}}
    (output / 'sample.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def tenth_table(labels, targets, tenth):
    """[query, tenth of own block (10), band, (n, missed)] for one build's returned IDs."""
    band_of = np.zeros(101, dtype=np.int64)
    for b, (lo, hi) in enumerate(BANDS):
        band_of[lo:hi + 1] = b
    missed = order_pairs.missed(labels, targets)
    table = np.zeros((len(targets), TENTHS, len(BANDS), 2), dtype=np.int64)
    ranks = band_of[np.arange(1, 101)]
    for q in range(len(targets)):
        np.add.at(table[q], (tenth[targets[q]], ranks, 0), 1)
        np.add.at(table[q], (tenth[targets[q]], ranks, 1), missed[q].astype(np.int64))
    return table


def _seeded(builds, name, ef, load):
    return {int(p.name.rsplit('-', 1)[1]): load(p) for p in sorted(Path(builds).glob(f'{name}-*'))
            if (p / f'labels-ef{ef}.npy').exists()}


def _difference(tables, draws, seed):
    """Within-query last-minus-first-tenth miss difference, seeds averaged within query, query bootstrap."""
    mean = np.mean(list(tables.values()), axis=0)
    rows = np.random.default_rng(seed).integers(0, mean.shape[0], (draws, mean.shape[0]))
    boot = np.array([risk_difference(mean[r], TENTHS, 1) for r in rows])
    return {'value_pp': 100 * risk_difference(mean, TENTHS, 1), 'interval_95_pp': (100 * np.nanquantile(boot, [.025, .975])).tolist(),
            'seeds': sorted(tables), 'per_seed_pp': {str(s): 100 * risk_difference(t, TENTHS, 1) for s, t in tables.items()}}


def within_block(sample, builds, name, *, ef=200, draws=5000, seed=20261124):
    sample = Path(sample)
    targets, tenth = np.load(sample / 'exact-top100.npy'), np.load(sample / 'block-tenth.npy')
    tables = _seeded(builds, name, ef, lambda p: tenth_table(np.load(p / f'labels-ef{ef}.npy'), targets, tenth))
    if not tables:
        raise ValueError(f'no {name} build with labels at ef {ef}')
    return _difference(tables, draws, seed)


def global_position(builds, *, ef=200, draws=5000, seed=20261124):
    """Shuffled builds: within-query difference between the last and first insertion tenths."""
    tables = _seeded(builds, 'shuffled', ef, lambda p: np.load(p / f'per-query-position-ef{ef}.npy'))
    if not tables:
        raise ValueError('no shuffled build')
    return _difference(tables, draws, seed)


def analyze(sample, order_builds, engine_builds, *, ef=200):
    out = {'kind': 'EXPLORATORY public streaming workload (aggregates only)', 'ef': ef,
           'sample': json.loads((Path(sample) / 'sample.json').read_text())}
    for backend, folder in sorted(order_builds.items()):
        paired = order_pairs.compare(sample, folder, 'appended', 'reversed', ef=ef)
        out[f'order_{backend}'] = {
            'primary_gap_change_appended_to_reversed': paired['gap_change_pp'],
            'paired_appended_to_reversed': paired,
            'within_block': {o: within_block(sample, folder, o, ef=ef) for o in ('appended', 'reversed')},
            'global_position_shuffled': global_position(folder, ef=ef)}
    if engine_builds is not None:
        block = {}
        for repair in ('repaired', 'repaired8'):
            effect = order_pairs.compare(sample, engine_builds, 'grown', repair, ef=ef)
            margins = engine_arm_analysis.analyze(engine_builds, ef=ef, arm=repair)
            block[repair] = {'effect_gap_change': effect, 'margins': margins,
                             'decision': {'effect': bool(effect['gap_change_pp']['interval_95'][1] < 0),
                                          'oldest_no_harm': margins['decision']['oldest_no_harm'],
                                          'recall_no_harm': margins['decision']['recall_no_harm']}}
        block['within_block'] = {arm: within_block(sample, engine_builds, arm, ef=ef)
                                 for arm in ('grown', 'repaired', 'repaired8', 'rebuilt')}
        out['engine'] = block
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('data', 'queries', 'gt', 'runbook', 'output'):
        p.add_argument(name, type=Path)
    a = sub.add_parser('analyze')
    a.add_argument('sample', type=Path)
    a.add_argument('order_builds', nargs='+', help='BACKEND=FOLDER of {order}-{seed} builds')
    a.add_argument('--engine', type=Path, help='folder of {arm}-{seed} engine builds')
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--ef', type=int, default=200)
    args = parser.parse_args()
    if args.command == 'prepare':
        summary = prepare(args.data, args.queries, args.gt, args.runbook, args.output)
        print(json.dumps({k: summary[k] for k in ('count', 'queries', 'cohort_sizes', 'ground_truth_check')}, indent=2))
    else:
        folders = dict(item.split('=', 1) for item in args.order_builds)
        out = analyze(args.sample, folders, args.engine, ef=args.ef)
        args.output.write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps({k: v for k, v in out.items() if k != 'sample'}, indent=2)[:4000])


if __name__ == '__main__':
    main()
