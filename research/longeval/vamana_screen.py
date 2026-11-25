"""Screen M3: arrival bias in a Vamana (FreshDiskANN-style) graph (design: docs/screen-m3-vamana.md,
recorded before this code was written).

    python -m research.longeval.vamana_screen run SAMPLE OUTPUT --order shuffled --seed 11
    python -m research.longeval.vamana_screen analyze OUT.json FOLDER

`run` writes the sample's vectors, queries and insertion order as raw files in OUTPUT, compiles
`vamana_driver.cpp` once (g++ -O3 -march=native), builds and searches, and scores the returned rows as
`insertion_pilot.run` does (`per-query-ef{L}.npy`, `report.json` with the in-degree by insertion decile).
`analyze` reports the within-query ratios of `insertion_strata.orders`. Aggregates only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .accumulation import miss_table
from .insertion_pilot import insertion_order
from .insertion_strata import orders

R, L_BUILD, ALPHA = 32, 100, 1.2
WIDTHS = (100, 200, 400, 800)
SOURCE = Path(__file__).with_name('vamana_driver.cpp')


def driver(cache):
    """The compiled driver, keyed by the source's hash so that a changed source is rebuilt."""
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    binary = Path(cache) / f'vamana_driver-{digest}'
    if not binary.exists():
        binary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['g++', '-O3', '-march=native', '-std=c++17', str(SOURCE), '-o', str(binary)], check=True)
    return binary


def write_matrix(path, array):
    array = np.ascontiguousarray(array, dtype='<f4')
    with open(path, 'wb') as f:
        np.array(array.shape, dtype='<u4').tofile(f)
        array.tofile(f)


def run(sample, output, *, order, seed, cache, widths=WIDTHS):
    sample, output = Path(sample), Path(output)
    vectors = np.load(sample / 'vectors.npy', mmap_mode='r')
    cohort = np.load(sample / 'cohort.npy')
    queries = np.load(sample / 'queries.npy')
    exact = np.load(sample / 'exact-top100.npy')
    metric = json.loads((sample / 'sample.json').read_text()).get('metric', 'ip')
    if metric == 'ip':
        norms = np.linalg.norm(np.asarray(vectors[:10_000], dtype=np.float64), axis=1)
        if np.abs(norms - 1).max() > 1e-4:
            raise ValueError('inner-product data must be unit-normalised for the Euclidean driver')
    n = len(vectors)
    sequence = insertion_order(np.load(sample / 'lexical-rank.npy'), cohort, order, seed)
    position = np.empty(n, dtype=np.int64)
    position[sequence] = np.arange(n)
    output.mkdir(parents=True, exist_ok=False)
    write_matrix(output / 'vectors.bin', vectors)
    write_matrix(output / 'queries.bin', queries)
    with open(output / 'order.bin', 'wb') as f:
        np.array([n], dtype='<u4').tofile(f)
        sequence.astype('<u4').tofile(f)
    result = subprocess.run([str(driver(cache)), str(output / 'vectors.bin'), str(output / 'order.bin'), str(output / 'queries.bin'),
                             str(output), str(R), str(L_BUILD), str(ALPHA), ','.join(map(str, widths))],
                            check=True, capture_output=True, text=True)
    timing = json.loads(result.stdout.strip().splitlines()[-1])
    for name in ('vectors.bin', 'queries.bin'):
        (output / name).unlink()
    indegree = np.fromfile(output / 'indegree.bin', dtype='<i4')
    report = {'backend': 'vamana', 'implementation': 'research/longeval/vamana_driver.cpp (independent, FreshDiskANN Algorithms 1-3)',
              'driver_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'order': order, 'seed': seed, 'ef': 200,
              'R': R, 'L_build': L_BUILD, 'alpha': ALPHA, 'count': int(n), 'metric': metric, **timing,
              'indegree_by_insertion_decile': [float(indegree[(position * 10 // n) == d].mean()) for d in range(10)],
              'indegree_zero_share': float((indegree == 0).mean()), 'widths': {}}
    for width in widths:
        found = np.fromfile(output / f'labels-L{width}.bin', dtype='<i4').reshape(len(queries), 100).astype(np.int64)
        table = miss_table(found, exact, cohort.astype(np.int64) - 1, 3)
        np.save(output / f'per-query-ef{width}.npy', table)
        report['widths'][str(width)] = {
            'recall_at_100': float(np.mean([np.isin(exact[q], found[q]).mean() for q in range(len(queries))])),
            'miss_rate': {str(c): float(table[:, c - 1, :, 1].sum() / table[:, c - 1, :, 0].sum()) for c in (1, 2, 3)}}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('order', 'seed', 'build_seconds', 'indegree_by_insertion_decile')} | {'recall_ef200': report['widths']['200']['recall_at_100']}))
    return report


def read_diskann_graph(folder, prefix):
    """In-degree per tag from a saved DynamicMemoryIndex: the graph file (uint64 size, uint32 max degree,
    uint32 entry, uint64 frozen points, then per location uint32 k and k uint32 locations) and the tags file
    (uint32 count, uint32 dims, then one uint32 tag per location; frozen points carry no tag)."""
    raw = (Path(folder) / prefix).read_bytes()
    size, _, _, frozen = np.frombuffer(raw[:24], dtype=np.dtype([('s', '<u8'), ('m', '<u4'), ('e', '<u4'), ('f', '<u8')]))[0]
    if size != len(raw):
        raise ValueError('graph file size differs from its header')
    words = np.frombuffer(raw[24:], dtype='<u4')
    lists, i = [], 0
    while i < len(words):
        k = int(words[i])
        lists.append(words[i + 1:i + 1 + k])
        i += 1 + k
    tags_raw = (Path(folder) / f'{prefix}.tags').read_bytes()
    count = int(np.frombuffer(tags_raw[:4], dtype='<u4')[0])
    tags = np.frombuffer(tags_raw[8:8 + 4 * count], dtype='<u4')
    return lists, tags, int(frozen)


def run_diskann(sample, output, *, order, seed, widths=WIDTHS):
    """The same build with DiskANN's own DynamicMemoryIndex (design addendum of docs/screen-m3-vamana.md)."""
    import diskannpy
    sample, output = Path(sample), Path(output)
    vectors = np.load(sample / 'vectors.npy', mmap_mode='r')
    cohort = np.load(sample / 'cohort.npy')
    queries = np.ascontiguousarray(np.load(sample / 'queries.npy'), dtype=np.float32)
    exact = np.load(sample / 'exact-top100.npy')
    metric = json.loads((sample / 'sample.json').read_text()).get('metric', 'ip')
    if metric == 'ip':
        norms = np.linalg.norm(np.asarray(vectors[:10_000], dtype=np.float64), axis=1)
        if np.abs(norms - 1).max() > 1e-4:
            raise ValueError('inner-product data must be unit-normalised for the Euclidean index')
    n, dim = vectors.shape
    sequence = insertion_order(np.load(sample / 'lexical-rank.npy'), cohort, order, seed)
    position = np.empty(n, dtype=np.int64)
    position[sequence] = np.arange(n)
    # Scratch sized for the widest search up front: growing it during a multi-threaded search corrupted the
    # heap in diskannpy 0.7.0 ("corrupted size vs. prev_size" after "Attempting to expand query scratch_space").
    index = diskannpy.DynamicMemoryIndex('l2', np.float32, dim, n + 1, L_BUILD, R, alpha=ALPHA, num_threads=1,
                                         initial_search_complexity=max(widths), search_threads=8)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    for label in sequence:
        index.insert(np.ascontiguousarray(vectors[label], dtype=np.float32), np.uint32(label + 1))
    build_seconds = time.perf_counter() - started
    index.save(str(output), 'graph')
    lists, tags, frozen = read_diskann_graph(output, 'graph')
    location_label = tags.astype(np.int64) - 1                    # location -> sample position; -1: no vector
    tagged = location_label >= 0                                    # unused slots and the frozen point carry tag 0
    # diskannpy 0.7.0's save leaves out the first inserted vector (location 0 holds no study vector and tag 0),
    # although the live index returns it; at most that one vector may be missing from the in-degree count.
    unmapped = n - int(tagged.sum())
    if unmapped > 1 or len(np.unique(location_label[tagged])) != tagged.sum():
        raise ValueError('the saved tags do not map the inserted vectors')
    indegree = np.zeros(n, dtype=np.int64)
    for location, links in enumerate(lists[:len(location_label)]):
        if not tagged[location]:
            continue
        links = links[links < len(location_label)].astype(np.int64)
        np.add.at(indegree, location_label[links[tagged[links]]], 1)
    for name in ('graph.data',):
        (output / name).unlink(missing_ok=True)
    report = {'backend': 'diskann', 'implementation': 'diskannpy 0.7.0 DynamicMemoryIndex', 'order': order, 'seed': seed,
              'ef': 200, 'R': R, 'L_build': L_BUILD, 'alpha': ALPHA, 'count': int(n), 'metric': metric,
              'build_seconds': build_seconds, 'frozen_points': frozen,
              'indegree_by_insertion_decile': [float(indegree[(position * 10 // n) == d].mean()) for d in range(10)],
              'indegree_zero_share': float((indegree == 0).mean()), 'saved_graph_unmapped_vectors': unmapped, 'widths': {}}
    for width in widths:
        found = np.asarray(index.batch_search(queries, 100, max(width, 100), 8).identifiers, dtype=np.int64) - 1
        table = miss_table(found, exact, cohort.astype(np.int64) - 1, 3)
        np.save(output / f'per-query-ef{width}.npy', table)
        report['widths'][str(width)] = {
            'recall_at_100': float(np.mean([np.isin(exact[q], found[q]).mean() for q in range(len(queries))])),
            'miss_rate': {str(c): float(table[:, c - 1, :, 1].sum() / table[:, c - 1, :, 0].sum()) for c in (1, 2, 3)}}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('order', 'seed', 'build_seconds', 'indegree_by_insertion_decile')} | {'recall_ef200': report['widths']['200']['recall_at_100']}))
    return report


def analyze(folder, ef=200):
    out = {'kind': 'EXPLORATORY screen M3 (docs/screen-m3-vamana.md), aggregates only', 'ef': ef,
           'orders': orders(folder, ef=ef), 'indegree_by_insertion_decile': {}}
    for path in sorted(Path(folder).glob('*/report.json')):
        r = json.loads(path.read_text())
        out['indegree_by_insertion_decile'][f"{r['order']}-{r['seed']}"] = r['indegree_by_insertion_decile']
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('sample', type=Path)
    r.add_argument('output', type=Path)
    r.add_argument('--order', choices=('appended', 'shuffled', 'reversed'), required=True)
    r.add_argument('--seed', type=int, required=True)
    r.add_argument('--cache', type=Path, default=Path.home() / '.cache' / 'ann-longeval-research')
    r.add_argument('--backend', choices=('driver', 'diskann'), default='driver')
    a = sub.add_parser('analyze')
    a.add_argument('output', type=Path)
    a.add_argument('folder', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        if args.backend == 'diskann':
            run_diskann(args.sample, args.output, order=args.order, seed=args.seed)
        else:
            run(args.sample, args.output, order=args.order, seed=args.seed, cache=args.cache)
    else:
        out = analyze(args.folder)
        args.output.write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps(out, indent=2)[:3000])


if __name__ == '__main__':
    main()
