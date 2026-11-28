"""Screen M2: arrival bias when documents are also deleted (design: docs/screen-m2-deletions.md,
recorded before this code was written).

    python -m research.longeval.deletion_screen prepare-bigann DATA.fbin QUERIES.fbin RUNBOOK.yaml OUTPUT
    python -m research.longeval.deletion_screen prepare-longeval SAMPLE OUTPUT --seed 11
    python -m research.longeval.deletion_screen run PREPARED OUTPUT --backend epzero --arm grown --seed 11
    python -m research.longeval.deletion_screen analyze OUT.json PREPARED BUILDS

`prepare-*` writes a schedule (insert ranges and deletions, in step order, over a fixed insertion
sequence), the arrival cohort of every vector, the final live set and the exact top 100 of the live
set. `run` replays the schedule in one arm: `grown` (deletions as tombstones), `rebuilt` (only the final
live set, in the same order, no deletions) or `repaired` (grown with extra_backlinks 8 and each insert
step refined at width 100; Epzero only). Builds are single-threaded; searches use every core. The
runbook is read as data with a small parser. Aggregates only.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from .accumulation import exact_prefix, miss_table
from .insertion_pilot import insertion_order
from .insertion_strata import risk_difference

EFS = (100, 200, 400)
K = 100
DRAWS, RNG = 5000, 20261126
REPAIR_LINKS, REPAIR_EF = 8, 100
LONGEVAL_DELETE_SHARE = 0.10


def parse_steps(text, dataset):
    """[(operation, start, end)] for inserts and deletes of one dataset's runbook, in step order."""
    steps, current, inside = {}, None, False
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
        elif current is not None and (m := re.fullmatch(r'\s+(operation|start|end)\s*:\s*["\']?(\w+)["\']?\s*', line)):
            current[m.group(1)] = m.group(2)
    out = []
    for key in sorted(steps):
        s = steps[key]
        if s.get('operation') in ('insert', 'delete'):
            out.append((s['operation'], int(s['start']), int(s['end'])))
        elif s.get('operation') != 'search':
            raise ValueError(f'unexpected operation {s.get("operation")}')
    return out


def check_schedule(steps, rows):
    live = np.zeros(rows, dtype=bool)
    added = np.zeros(rows, dtype=bool)
    for op, start, end in steps:
        if op == 'insert':
            if added[start:end].any():
                raise ValueError('a vector is inserted twice')
            added[start:end] = live[start:end] = True
        else:
            if not live[start:end].all():
                raise ValueError('a deletion removes a vector that is not live')
            live[start:end] = False
    if not added.all():
        raise ValueError('the schedule does not insert every vector')
    return live


def _write(output, sequence, steps, cohort, live, queries, exact, meta):
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'sequence.npy', sequence.astype(np.int64))
    np.save(output / 'cohort.npy', cohort.astype(np.int8))
    np.save(output / 'live.npy', live)
    np.save(output / 'queries.npy', np.ascontiguousarray(queries, dtype=np.float32))
    np.save(output / 'exact-top100.npy', exact.astype(np.int64))
    meta = meta | {'steps': [[op, int(a), int(b)] for op, a, b in steps], 'live': int(live.sum()),
                   'cohort_live': {str(c): int(((cohort == c) & live).sum()) for c in (1, 2, 3)}}
    (output / 'prepared.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


def prepare_bigann(data, queries, runbook, output, dataset='msturing-10M-clustered'):
    """Steps index positions of the file order, which is the insertion order."""
    from .bigann_stream import read_bin, sha256
    vectors, q = read_bin(data), read_bin(queries)
    steps = parse_steps(Path(runbook).read_text(), dataset)
    live = check_schedule(steps, len(vectors))
    first, second = (len(vectors) + 2) // 3, (2 * len(vectors) + 2) // 3
    cohort = np.empty(len(vectors), dtype=np.int8)
    for op, start, end in steps:
        if op == 'insert':
            cohort[start:end] = 1 if start < first else 2 if start < second else 3
    exact = exact_prefix(vectors, q, np.flatnonzero(live), 'l2', chunk=10_000)   # 10,000 queries: smaller scan blocks
    meta = {'kind': 'bigann', 'dataset': dataset, 'data': str(data), 'metric': 'l2', 'count': int(len(vectors)),
            'dimension': int(vectors.shape[1]), 'queries': int(len(q)), 'runbook_sha256': sha256(runbook)}
    return _write(Path(output), np.arange(len(vectors)), steps, cohort, live, q, exact, meta)


def prepare_longeval(sample, output, *, seed):
    """Cohort 1 in lexical order, then before cohorts 2 and 3 a random 10% of the live documents tombstoned.
    Steps index positions of the insertion sequence."""
    sample = Path(sample)
    vectors = np.load(sample / 'vectors.npy', mmap_mode='r')
    cohort = np.load(sample / 'cohort.npy')
    q = np.load(sample / 'queries.npy')
    sequence = insertion_order(np.load(sample / 'lexical-rank.npy'), cohort, 'appended', 0)
    rng = np.random.default_rng(seed)
    bounds = np.cumsum([0] + [int((cohort == c).sum()) for c in (1, 2, 3)])
    steps, live_pos = [], np.zeros(len(sequence), dtype=bool)
    for c in range(3):
        if c > 0:
            candidates = np.flatnonzero(live_pos)
            chosen = np.sort(rng.choice(candidates, int(round(LONGEVAL_DELETE_SHARE * len(candidates))), replace=False))
            for p in chosen:
                steps.append(('delete', int(p), int(p) + 1))
            live_pos[chosen] = False
        steps.append(('insert', int(bounds[c]), int(bounds[c + 1])))
        live_pos[bounds[c]:bounds[c + 1]] = True
    live = np.zeros(len(sequence), dtype=bool)
    live[sequence[live_pos]] = True                      # by sample position
    exact = exact_prefix(vectors, q, np.flatnonzero(live), 'ip')
    meta = {'kind': 'longeval', 'data': str(sample / 'vectors.npy'), 'metric': 'ip', 'count': int(len(sequence)),
            'dimension': int(vectors.shape[1]), 'queries': int(len(q)), 'delete_seed': seed,
            'deleted': int((~live_pos).sum())}
    return _write(Path(output), sequence, steps, cohort, live, q, exact, meta)


def _vectors(prepared, meta):
    if meta['kind'] == 'bigann':
        from .bigann_stream import read_bin
        return read_bin(meta['data'])
    return np.load(meta['data'], mmap_mode='r')


def run(prepared, output, *, backend, arm, seed, M=16, ef_construction=100, threads=0, batch=1024):
    prepared, output = Path(prepared), Path(output)
    meta = json.loads((prepared / 'prepared.json').read_text())
    vectors = _vectors(prepared, meta)
    sequence = np.load(prepared / 'sequence.npy')
    live = np.load(prepared / 'live.npy')
    queries = np.load(prepared / 'queries.npy')
    exact = np.load(prepared / 'exact-top100.npy')
    cohort = np.load(prepared / 'cohort.npy')
    n, dim = meta['count'], meta['dimension']
    repairing = arm in ('repaired', 'rebuilt-repaired')          # screen A5 adds the repaired rebuild
    rebuilding = arm in ('rebuilt', 'rebuilt-repaired')
    if repairing and backend != 'epzero':
        raise ValueError('the repair is an Epzero arm')
    if backend == 'epzero':
        import epzero
        index = epzero.Index(dim=dim, metric=meta['metric'], max_elements=n, M=M, ef_construction=ef_construction, seed=seed,
                             quantization='none', keep_vectors=True, nav_dim=0, fill_links=False,
                             fast_build=False, half_vectors=False)
        if repairing:
            index.set_extra_backlinks(REPAIR_LINKS)

        def add(labels):
            for s in range(0, len(labels), batch):
                chunk = labels[s:s + batch]
                index.add(np.ascontiguousarray(vectors[chunk], dtype=np.float32), labels=chunk.astype(np.int64), threads=1)

        def delete(labels):
            for label in labels:
                if not index.remove(int(label)):
                    raise ValueError('a deleted label was not live')

        def search(width):
            found, _ = index.search(queries, k=K, ef=width, rerank=0, rerank_z=0, patience=0, threads=threads)
            return np.asarray(found, dtype=np.int64)
        version = epzero.__version__
    elif backend == 'hnswlib':
        import hnswlib
        index = hnswlib.Index(space='l2' if meta['metric'] == 'l2' else 'ip', dim=dim)
        index.init_index(max_elements=n, M=M, ef_construction=ef_construction, random_seed=seed)

        def add(labels):
            for s in range(0, len(labels), batch):
                chunk = labels[s:s + batch]
                index.add_items(np.ascontiguousarray(vectors[chunk], dtype=np.float32), chunk.astype(np.int64), num_threads=1)

        def delete(labels):
            for label in labels:
                index.mark_deleted(int(label))

        def search(width):
            index.set_ef(width)
            found, _ = index.knn_query(queries, k=K, num_threads=max(threads, 1) if threads else -1)
            return np.asarray(found, dtype=np.int64)
        version = getattr(hnswlib, '__version__', 'unknown')
    else:
        raise ValueError('backend must be epzero or hnswlib')
    started, added, refine_seconds = time.perf_counter(), 0, 0.0
    for op, start, end in meta['steps']:
        labels = sequence[start:end]
        if op == 'insert':
            if rebuilding:
                labels = labels[live[labels]]
            add(labels)
            if repairing and added > 0:
                t0 = time.perf_counter()
                index.refine_links(threads=threads, ef=REPAIR_EF, first=added)
                refine_seconds += time.perf_counter() - t0
            added += len(labels)
        elif not rebuilding:
            delete(labels)
    build_seconds = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    widths = {}
    for width in EFS:
        found = search(width)
        if not live[found[found >= 0]].all():
            raise ValueError('a search returned a deleted vector')
        table = miss_table(found, exact, cohort.astype(np.int64) - 1, 3)
        np.save(output / f'per-query-ef{width}.npy', table)
        recall = float(np.mean([np.isin(exact[q], found[q]).mean() for q in range(len(exact))]))
        widths[str(width)] = {'recall_at_100': recall,
                              'miss_rate': {str(c): float(table[:, c - 1, :, 1].sum() / table[:, c - 1, :, 0].sum()) for c in (1, 2, 3)},
                              'excess_pp': 100 * risk_difference(table, 3, 1)}
    report = {'backend': backend, 'version': version, 'arm': arm, 'seed': seed, 'M': M, 'ef_construction': ef_construction,
              'prepared': meta.get('dataset', meta['kind']), 'build_seconds': build_seconds, 'refine_seconds': refine_seconds,
              'widths': widths}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


def _tables(builds, backend, arm, ef):
    found = {}
    prefix = f'{backend}-{arm}-'
    for path in sorted(Path(builds).glob(prefix + '*')):
        if not path.name[len(prefix):].isdigit():               # 'rebuilt-' must not match 'rebuilt-repaired-'
            continue
        if (path / f'per-query-ef{ef}.npy').exists():
            found[int(path.name.rsplit('-', 1)[1])] = np.load(path / f'per-query-ef{ef}.npy')
    return found


def _mean(tables, seeds):
    return np.mean([tables[s] for s in seeds], axis=0)


def _oldest(table):
    return table[:, 0, :, 1].sum() / table[:, 0, :, 0].sum()


def analyze(prepared, builds, *, ef=200):
    meta = json.loads((Path(prepared) / 'prepared.json').read_text())
    out = {'kind': 'EXPLORATORY screen M2 (docs/screen-m2-deletions.md), aggregates only', 'ef': ef,
           'prepared': {k: v for k, v in meta.items() if k not in ('steps', 'data')}, 'backends': {}}
    rng = np.random.default_rng(RNG)
    queries = meta['queries']
    draws = rng.integers(0, queries, (DRAWS, queries))
    for backend in ('epzero', 'hnswlib'):
        arms = {arm: _tables(builds, backend, arm, ef) for arm in ('grown', 'rebuilt', 'repaired', 'rebuilt-repaired')}
        if not arms['grown']:
            continue
        block = {'arms': {}, 'contrasts': {}}
        for arm, tables in arms.items():
            if not tables:
                continue
            mean = _mean(tables, sorted(tables))
            boot = np.array([risk_difference(mean[r], 3, 1) for r in draws])
            block['arms'][arm] = {'seeds': sorted(tables), 'excess_pp': 100 * risk_difference(mean, 3, 1),
                                  'excess_interval_95_pp': (100 * np.quantile(boot, [.025, .975])).tolist(),
                                  'oldest_miss_rate': float(_oldest(mean))}
        for other, base in (('rebuilt', 'grown'), ('repaired', 'grown'), ('rebuilt-repaired', 'grown'),
                            ('rebuilt-repaired', 'rebuilt')):
            seeds = sorted(set(arms[base]) & set(arms[other]))
            if not seeds:
                continue
            a, b = _mean(arms[other], seeds), _mean(arms[base], seeds)
            diff = 100 * (risk_difference(a, 3, 1) - risk_difference(b, 3, 1))
            boot = np.array([100 * (risk_difference(a[r], 3, 1) - risk_difference(b[r], 3, 1)) for r in draws])
            old = np.array([100 * (_oldest(a[r]) - _oldest(b[r])) for r in draws])
            block['contrasts'][f'{other}_minus_{base}'] = {
                'seeds': seeds, 'excess_change_pp': diff, 'interval_95_pp': np.quantile(boot, [.025, .975]).tolist(),
                'oldest_miss_change_pp': 100 * float(_oldest(a) - _oldest(b)), 'oldest_upper_95_pp': float(np.quantile(old, .975))}
        out['backends'][backend] = block
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare-bigann')
    for name in ('data', 'queries', 'runbook', 'output'):
        p.add_argument(name, type=Path)
    q = sub.add_parser('prepare-longeval')
    q.add_argument('sample', type=Path)
    q.add_argument('output', type=Path)
    q.add_argument('--seed', type=int, required=True)
    r = sub.add_parser('run')
    r.add_argument('prepared', type=Path)
    r.add_argument('output', type=Path)
    r.add_argument('--backend', choices=('epzero', 'hnswlib'), required=True)
    r.add_argument('--arm', choices=('grown', 'rebuilt', 'repaired', 'rebuilt-repaired'), required=True)
    r.add_argument('--seed', type=int, required=True)
    r.add_argument('--threads', type=int, default=0)
    a = sub.add_parser('analyze')
    a.add_argument('output', type=Path)
    a.add_argument('prepared', type=Path)
    a.add_argument('builds', type=Path)
    a.add_argument('--ef', type=int, default=200)
    args = parser.parse_args()
    if args.command == 'prepare-bigann':
        print(json.dumps({k: v for k, v in prepare_bigann(args.data, args.queries, args.runbook, args.output).items() if k != 'steps'}, indent=2))
    elif args.command == 'prepare-longeval':
        print(json.dumps({k: v for k, v in prepare_longeval(args.sample, args.output, seed=args.seed).items() if k != 'steps'}, indent=2))
    elif args.command == 'run':
        run(args.prepared, args.output, backend=args.backend, arm=args.arm, seed=args.seed, threads=args.threads)
    else:
        out = analyze(args.prepared, args.builds, ef=args.ef)
        args.output.write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps(out['backends'], indent=2))


if __name__ == '__main__':
    main()
