"""EXPLORATORY pilot: does HNSW's miss rate follow insertion position? (post-freeze, sample scale)

On a deterministic sample of snapshot-3 documents that keeps each document's arrival snapshot,
build a single-threaded HNSW index (FAISS or Epzero; M=16, efConstruction=100, inner product) in one of three
insertion orders and compare, per arrival cohort:

  appended   cohort 1, then 2, then 3 (lexical within cohort), as the collection actually grew
  reversed   cohort 3, then 2, then 1
  shuffled   one random permutation

Measured at ef 200 with the 381 test queries: neighbour recall against exact search within the
sample, misses by cohort within exact-rank bands, and base-layer in-degree by cohort. If misses
follow insertion position, the reversed build must move the excess onto cohort 1.

A build can also be searched at further widths (`--efs`) and built with other graph settings
(`--M`, `--ef-construction`), to check that the result does not rest on one setting. Every build
also records its misses by insertion decile. In a shuffled build the decile is independent of the
document, so that table isolates the effect of position.

`prepare-public` puts a public benchmark into the same format with arrival groups assigned at
random. Position is then the only thing that differs between the groups.

`prepare-full` prepares the full-scale experiment in the same format: all of snapshot 3, or a
draw of snapshot-1 size that keeps the arrival groups in proportion. The builds are then run by
the same `run` command (scripts/pace/LONGEVAL_INSERTION_ORDER.md).

This is a mechanism pilot on a sample, not a result about the full collection.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

BANDS = ((1, 10), (11, 50), (51, 100))


def prepare(dense, inputs, output, *, count=300_000):
    """Sample documents, store their vectors, cohorts and the exact top-100 of every test query."""
    from .dense import ARTIFACTS
    from .heldout import ROLE, load_heldout
    from .heldout_exploratory import cohorts
    from .mapped import MappedSnapshot
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    first = cohorts(dense)
    with MappedSnapshot(Path(dense) / 'snapshot-3', expected_archive_sha256=ARTIFACTS[2]['archive_sha256']) as snap:
        order = sorted(range(len(snap.ids)), key=lambda i: hashlib.sha256(b'insertion-pilot-v1:' + snap.ids[i].encode()).digest())
        rows = np.sort(np.array(order[:count]))
        ids = [snap.ids[i] for i in rows]
        vectors = np.empty((count, snap.vectors.shape[1]), dtype=np.float32)
        for start in range(0, count, 4096):
            vectors[start:start + 4096] = snap.vectors[rows[start:start + 4096]]
    cohort = np.array([first[d] for d in ids], dtype=np.int8)
    queries, _, _, _ = load_heldout(inputs, references=False)
    q = np.ascontiguousarray(queries[ROLE].vectors)
    scores = np.empty((len(q), count), dtype=np.float32)
    q64 = q.astype(np.float64)
    for start in range(0, count, 20_000):
        scores[:, start:start + 20_000] = (vectors[start:start + 20_000].astype(np.float64) @ q64.T).T
    top = np.argsort(-scores, axis=1, kind='stable')[:, :100].astype(np.int32)
    np.save(output / 'vectors.npy', vectors)
    np.save(output / 'cohort.npy', cohort)
    # Lexical rank of each sampled ID: the study inserts each cohort's additions in lexical ID order.
    lexical = np.empty(count, dtype=np.int64)
    lexical[np.argsort(np.array(ids, dtype=object), kind='stable')] = np.arange(count)
    np.save(output / 'lexical-rank.npy', lexical)
    np.save(output / 'queries.npy', q)
    np.save(output / 'exact-top100.npy', top)
    summary = {'count': count, 'cohort_sizes': {str(c): int((cohort == c).sum()) for c in (1, 2, 3)},
               'queries': len(q), 'ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest()}
    (output / 'sample.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


DRAW_SIZE = 869_902                    # the size of snapshot 1


def quotas(sizes, count):
    """Split `count` over groups in proportion to their sizes (largest remainders get the leftover)."""
    total = sum(sizes)
    if not 0 < count <= total:
        raise ValueError('draw size must be positive and at most the collection size')
    shares = [count * size / total for size in sizes]
    out = [int(share) for share in shares]
    for index in sorted(range(len(sizes)), key=lambda i: (out[i] - shares[i], i))[:count - sum(out)]:
        out[index] += 1
    return out


def prepare_full(dense, inputs, output, *, draw=0, count=None):
    """The full-scale sample: all of snapshot 3 (draw 0), or draw 1, 2, ... of `count` documents
    (default: the size of snapshot 1) that keeps the three arrival groups in proportion.

    Exact neighbours are float64 inner products ordered by descending score, then descending
    document ID, as in the study's exact search. For draw 0 they are compared with the held-out
    exact references under the study's own rule: two exhaustive runs may differ only by rounding
    near ties (heldout.reference_agreement).
    """
    from .dense import ARTIFACTS
    from .heldout import ROLE, _development_reference, load_heldout, reference_agreement
    from .heldout_exploratory import cohorts
    from .mapped import MappedSnapshot
    from .query_diagnostic import load_queries
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    first = cohorts(dense)
    queries, refs, _, _ = load_heldout(inputs)
    query_ids = list(queries[ROLE].ids)
    test_vectors = np.ascontiguousarray(queries[ROLE].vectors)
    confirm_ids, confirm_vectors = [], np.empty((0, test_vectors.shape[1]), dtype=np.float32)
    if draw == 0:
        # The 100 training queries: never used for any arrival-order outcome, so they confirm it.
        training = Path(inputs) / 'training'
        confirm_ids, values, raw = load_queries(training / 'queries', training / 'float32')
        if raw['dtype'] != 'float32' or raw['device'] != 'cuda' or raw.get('tf32') is not False:
            raise ValueError('confirmation queries must use the declared CUDA float32 representation')
        if set(confirm_ids) & set(query_ids):
            raise ValueError('confirmation and test query IDs overlap')
        confirm_ids, confirm_vectors = list(confirm_ids), np.ascontiguousarray(values, dtype=np.float32)
    q64 = np.vstack([test_vectors, confirm_vectors]).astype(np.float64)
    with MappedSnapshot(Path(dense) / 'snapshot-3', expected_archive_sha256=ARTIFACTS[2]['archive_sha256']) as snap:
        cohort_all = np.array([first[doc] for doc in snap.ids], dtype=np.int8)
        if draw == 0:
            if count is not None:
                raise ValueError('draw 0 is the whole snapshot')
            rows = np.arange(len(snap.ids))
        else:
            wanted = quotas([int((cohort_all == c).sum()) for c in (1, 2, 3)], DRAW_SIZE if count is None else count)
            chosen = []
            for c, quota in zip((1, 2, 3), wanted):
                members = np.flatnonzero(cohort_all == c)
                keys = sorted(members.tolist(), key=lambda i: hashlib.sha256(
                    f'insertion-draw-v1:{draw}:'.encode() + snap.ids[i].encode()).digest())
                chosen.extend(keys[:quota])
            rows = np.sort(np.array(chosen))
        n, dim = len(rows), snap.vectors.shape[1]
        ids = [snap.ids[i] for i in rows]
        vectors = np.lib.format.open_memmap(output / 'vectors.npy', mode='w+', dtype=np.float32, shape=(n, dim))
        scores = np.empty((len(q64), n), dtype=np.float64)
        for start in range(0, n, 20_000):
            block = np.asarray(snap.vectors[rows[start:start + 20_000]], dtype=np.float32)
            vectors[start:start + len(block)] = block
            scores[:, start:start + len(block)] = (block.astype(np.float64) @ q64.T).T
        vectors.flush()
        del vectors
    cohort = cohort_all[rows]
    lexical = np.empty(n, dtype=np.int64)
    lexical[np.argsort(np.array(ids, dtype=object), kind='stable')] = np.arange(n)
    # Top 100 by descending score, then descending document ID; candidates are taken generously around the cut.
    top = np.empty((len(q64), 100), dtype=np.int32)
    top_scores = np.empty((len(q64), 100), dtype=np.float64)
    for qi in range(len(q64)):
        row = scores[qi]
        candidates = np.argpartition(-row, 255)[:256]
        candidates = candidates[np.lexsort((-lexical[candidates], -row[candidates]))]
        top[qi] = candidates[:100]
        top_scores[qi] = row[top[qi]]
    del scores
    nt = len(query_ids)
    np.save(output / 'cohort.npy', cohort)
    np.save(output / 'lexical-rank.npy', lexical)
    np.save(output / 'queries.npy', test_vectors)
    np.save(output / 'exact-top100.npy', top[:nt])
    (output / 'query-ids.json').write_text(json.dumps(query_ids) + '\n')
    summary = {'count': n, 'cohort_sizes': {str(c): int((cohort == c).sum()) for c in (1, 2, 3)},
               'queries': len(query_ids), 'metric': 'ip', 'draw': draw, 'collection': 'snapshot 3',
               'documents_in_snapshot': int(len(cohort_all)), 'ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest()}
    if draw == 0:
        # Document IDs in sample order, so returned positions can be mapped to LongEval documents.
        with gzip.open(output / 'doc-ids.txt.gz', 'wt', encoding='utf-8') as handle:
            handle.write('\n'.join(ids) + '\n')
        np.save(output / 'confirm-queries.npy', confirm_vectors)
        np.save(output / 'confirm-exact-top100.npy', top[nt:])
        (output / 'confirm-query-ids.json').write_text(json.dumps(confirm_ids) + '\n')
        summary['confirm_queries'] = len(confirm_ids)

        # The study's exact references for snapshot 3 must agree, up to rounding near ties.
        def listed(rows, offset):
            return {query: [(ids[i], float(score)) for i, score in zip(top[offset + qi], top_scores[offset + qi])]
                    for qi, query in enumerate(rows)}
        theirs = {query: [(doc, float(score)) for doc, score in refs[2][query][:100]] for query in query_ids}
        summary['heldout_reference_check'] = reference_agreement(listed(query_ids, 0), theirs)
        development = _development_reference(inputs, 3)[1]
        theirs = {query: [(doc, float(score)) for doc, score in development[query][:100]] for query in confirm_ids}
        summary['confirm_reference_check'] = reference_agreement(listed(confirm_ids, nt), theirs)
    (output / 'sample.json').write_text(json.dumps(summary, indent=2) + '\n')
    if draw == 0 and not (summary['heldout_reference_check']['passed'] and summary['confirm_reference_check']['passed']):
        raise ValueError('exact neighbours differ from the study references beyond rounding near ties')
    return summary


def compare_heldout(sample, build, baseline):
    """Neighbour recall per query of a chronological full-snapshot build at ef 200 against the held-out
    incremental run of the same implementation and seed. Recorded, not required: the two were built by
    different code, and only the insertion order and the seed are the same."""
    sample, build = Path(sample), Path(build)
    report = json.loads((build / 'report.json').read_text())
    baseline = json.loads(Path(baseline).read_text())
    if (baseline['backend'], baseline['seed'], baseline['mode']) != (report['backend'], report['seed'], 'incremental'):
        raise ValueError('the baseline is another arm')
    if report['order'] != 'appended' or report['ef'] != 200:
        raise ValueError('the comparison needs the chronological build at ef 200')
    theirs = baseline['snapshots'][2]['runs']['frozen']['ann_recall_per_query']
    table = np.load(build / 'per-query.npy')
    mine = 1 - table[:, :, :, 1].sum(axis=(1, 2)) / table[:, :, :, 0].sum(axis=(1, 2))
    other = np.array([theirs[query] for query in json.loads((sample / 'query-ids.json').read_text())])
    result = {'backend': report['backend'], 'seed': report['seed'], 'queries': int(len(mine)),
              'mean_recall_this_build': float(mine.mean()), 'mean_recall_heldout_run': float(other.mean()),
              'identical_per_query': bool(np.array_equal(np.round(mine, 6), np.round(other, 6))),
              'queries_that_differ': int(np.sum(np.round(mine, 6) != np.round(other, 6))),
              'largest_absolute_difference': float(np.abs(mine - other).max())}
    (build / 'heldout-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def collect(jobs, dataset, target):
    """Gather the builds of one dataset from returned job folders into `target/<backend>/<order>-<seed>`.

    Every copied file is checked against the job's own checksum list, and a job that did not end
    with exit code 0 is refused. `target/sample/sample.json` is the sample summary of the dataset.
    """
    import shutil
    jobs, target = Path(jobs), Path(target)
    target.mkdir(parents=True, exist_ok=False)
    summary, samples = [], set()
    for job in sorted(path for path in jobs.iterdir() if path.is_dir()):
        arm = dict(item.split('=') for item in (job / 'arm.txt').read_text().split())
        if arm['dataset'] != dataset:
            continue
        if (job / 'exit-code.txt').read_text().strip() != '0':
            raise ValueError(f'job {job.name} did not complete')
        sums = {name: digest for digest, name in
                (line.split(maxsplit=1) for line in (job / 'EVIDENCE-SHA256SUMS').read_text().splitlines())}
        root = job / dataset / arm['backend']
        for build in sorted(path for path in root.iterdir() if path.is_dir()):
            destination = target / arm['backend'] / build.name
            if destination.exists():
                raise ValueError(f'two jobs returned {arm["backend"]}/{build.name}')
            for path in sorted(build.iterdir()):
                listed = sums.get('./' + path.relative_to(job).as_posix())
                if listed is None or hashlib.sha256(path.read_bytes()).hexdigest() != listed:
                    raise ValueError(f'checksum mismatch or unlisted file: {path}')
            shutil.copytree(build, destination)
            summary.append({'job': job.name, 'backend': arm['backend'], 'seed': int(arm['seed']), 'build': build.name})
        samples.add((job / 'sample.json').read_text())
    if not summary:
        raise ValueError(f'no completed job for dataset {dataset}')
    if len(samples) != 1:
        raise ValueError('the jobs of one dataset disagree about the sample')
    (target / 'sample').mkdir()
    (target / 'sample' / 'sample.json').write_text(samples.pop())
    (target / 'collected.json').write_text(json.dumps({'dataset': dataset, 'builds': summary}, indent=2) + '\n')
    return {'dataset': dataset, 'builds': len(summary), 'jobs': sorted({item['job'] for item in summary})}


def read_fvecs(path):
    """Vectors of an .fvecs file (each row: int32 dimension, then float32 values), memory-mapped."""
    dim = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    data = np.memmap(path, dtype=np.float32, mode='r')
    if dim <= 0 or data.size % (dim + 1):
        raise ValueError('not an fvecs file')
    return data.reshape(-1, dim + 1)[:, 1:]


def prepare_public(base, queries, output, *, metric, count=300_000, query_count=1_000, seed=20261121,
                   cohort_sizes=(156_741, 81_467, 61_792)):
    """A public benchmark in the pilot's format, with arrival groups assigned at random.

    The groups have the pilot's sizes but no relation to the vectors, so any difference between them
    in a build comes from where they are inserted. `metric` is 'cosine' (vectors are normalized and
    searched by inner product) or 'l2'.
    """
    if metric not in ('cosine', 'l2'):
        raise ValueError('metric must be cosine or l2')
    data, q = read_fvecs(base), read_fvecs(queries)
    if sum(cohort_sizes) != count or count > len(data) or query_count > len(q) or data.shape[1] != q.shape[1]:
        raise ValueError('sample size, group sizes and files do not fit together')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.permutation(len(data))[:count])
    vectors = np.ascontiguousarray(data[rows], dtype=np.float32)
    q = np.ascontiguousarray(q[:query_count], dtype=np.float32)
    if metric == 'cosine':
        norms, query_norms = np.linalg.norm(vectors, axis=1, keepdims=True), np.linalg.norm(q, axis=1, keepdims=True)
        if not norms.all() or not query_norms.all():
            raise ValueError('a zero vector cannot be normalized')
        vectors /= norms
        q /= query_norms
    cohort = np.empty(count, dtype=np.int8)
    cohort[rng.permutation(count)] = np.repeat(np.array([1, 2, 3], dtype=np.int8), cohort_sizes)
    v64 = vectors.astype(np.float64)
    squared = (v64 ** 2).sum(axis=1)
    top = np.empty((len(q), 100), dtype=np.int32)
    for start in range(0, len(q), 100):
        scores = q[start:start + 100].astype(np.float64) @ v64.T
        if metric == 'l2':
            scores = 2 * scores - squared                    # larger is nearer; the query norm is constant per row
        top[start:start + 100] = np.argsort(-scores, axis=1, kind='stable')[:, :100]
    np.save(output / 'vectors.npy', vectors)
    np.save(output / 'cohort.npy', cohort)
    np.save(output / 'lexical-rank.npy', np.arange(count, dtype=np.int64))
    np.save(output / 'queries.npy', q)
    np.save(output / 'exact-top100.npy', top)
    summary = {'count': count, 'cohort_sizes': {str(c): int((cohort == c).sum()) for c in (1, 2, 3)},
               'queries': len(q), 'metric': 'ip' if metric == 'cosine' else 'l2', 'dimension': int(vectors.shape[1]),
               'source': {'base': Path(base).name, 'queries': Path(queries).name, 'vectors_in_file': int(len(data)),
                          'similarity': metric}, 'arrival_groups': 'assigned at random', 'seed': seed,
               'rows_sha256': hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest()}
    (output / 'sample.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def insertion_order(ids_sorted_rank, cohort, order, seed):
    """Insertion sequence (sample positions) for an order; cohorts are filled in lexical ID order."""
    base = np.lexsort((ids_sorted_rank, cohort))               # cohort, then lexical
    if order == 'appended':
        return base
    if order == 'reversed':
        return np.lexsort((ids_sorted_rank, -cohort.astype(np.int64)))
    if order == 'shuffled':
        return np.random.default_rng(seed).permutation(len(cohort))
    raise ValueError('unknown order')


def run(sample, output, *, order, seed, backend='faiss_hnsw', ef=200, M=16, ef_construction=100, efs=()):
    """Build one index and score it at `ef`; `efs` lists further search widths for the same build.

    Every width also saves the returned top-100 sample positions (`labels-ef{w}.npy`), so the same
    (query, target) pair can be followed across builds. If the sample holds a confirmation query set
    (`confirm-queries.npy`, `confirm-exact-top100.npy`), it is searched at the same widths and saved
    under the `confirm-` prefix; it is never used to choose anything.
    """
    import faiss
    faiss.omp_set_num_threads(1)
    if backend not in ('faiss_hnsw', 'epzero'):
        raise ValueError('unknown backend')
    sample, output = Path(sample), Path(output)
    vectors = np.load(sample / 'vectors.npy', mmap_mode='r')
    cohort = np.load(sample / 'cohort.npy')
    queries = np.load(sample / 'queries.npy')
    exact = np.load(sample / 'exact-top100.npy')
    confirm = None
    if (sample / 'confirm-queries.npy').exists():
        confirm = (np.load(sample / 'confirm-queries.npy'), np.load(sample / 'confirm-exact-top100.npy'))
    n, dim = vectors.shape
    metric = json.loads((sample / 'sample.json').read_text()).get('metric', 'ip')
    if metric not in ('ip', 'l2'):
        raise ValueError('unknown metric')
    sequence = insertion_order(np.load(sample / 'lexical-rank.npy'), cohort, order, seed)
    position = np.empty(n, dtype=np.int64)
    position[sequence] = np.arange(n)                       # insertion position of each sample document
    started = time.perf_counter()
    if backend == 'faiss_hnsw':
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2 if metric == 'l2' else faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.rng = faiss.RandomGenerator(seed)
        for start in range(0, n, 1024):
            block = np.ascontiguousarray(vectors[sequence[start:start + 1024]])
            for row in block:                               # one row per call keeps the order exact
                index.add(row.reshape(1, -1))
        build_seconds = time.perf_counter() - started

        def search(width, rows=queries):
            index.hnsw.efSearch = width
            _, labels = index.search(np.ascontiguousarray(rows), 100)
            return sequence[labels]                         # storage position -> sample position
        hnsw = index.hnsw
        neighbors = faiss.vector_to_array(hnsw.neighbors)
        offsets = faiss.vector_to_array(hnsw.offsets)
        width = hnsw.nb_neighbors(0)
        base = np.stack([neighbors[offsets[i]:offsets[i] + width] for i in range(n)])
        indegree = np.empty(n, dtype=np.int64)
        indegree[sequence] = np.bincount(base[base >= 0], minlength=n)
        outdegree = np.empty(n, dtype=np.int64)
        outdegree[sequence] = (base >= 0).sum(axis=1)
    else:
        # Same explicit, uncompressed configuration as the study's Epzero adapter (backends.py).
        import epzero
        index = epzero.Index(dim=dim, metric=metric, max_elements=n, M=M, ef_construction=ef_construction, seed=seed,
                             quantization='none', keep_vectors=True, nav_dim=0, fill_links=False,
                             fast_build=False, half_vectors=False)
        for start in range(0, n, 1024):
            chunk = sequence[start:start + 1024]
            index.add(np.ascontiguousarray(vectors[chunk]), labels=chunk.astype(np.int64), threads=1)
        build_seconds = time.perf_counter() - started

        def search(width, rows=queries):
            labels, _ = index.search(np.ascontiguousarray(rows), k=100, ef=width, rerank=0, rerank_z=0, patience=0,
                                     threads=1)
            return np.asarray(labels, dtype=np.int64)       # labels are sample positions
        indegree = np.zeros(n, dtype=np.int64)
        outdegree = np.zeros(n, dtype=np.int64)
        for label in range(n):
            links = np.asarray(index.neighbors(label, 0), dtype=np.int64)
            outdegree[label] = len(links)
            np.add.at(indegree, links, 1)
    def score(found, targets=exact):
        # Misses by cohort and exact-rank band, per query (for a query bootstrap).
        table = np.zeros((len(targets), 3, len(BANDS), 2), dtype=np.int64)   # [query, cohort, band, (n, missed)]
        place = np.zeros((len(targets), 10, len(BANDS), 2), dtype=np.int64)  # [query, insertion decile, band, (n, missed)]
        for qi in range(len(targets)):
            returned = set(found[qi].tolist())
            for rank, doc in enumerate(targets[qi], 1):
                band = next(i for i, b in enumerate(BANDS) if b[0] <= rank <= b[1])
                missed = doc not in returned
                table[qi, cohort[doc] - 1, band, 0] += 1
                table[qi, cohort[doc] - 1, band, 1] += missed
                place[qi, decile[doc], band, 0] += 1
                place[qi, decile[doc], band, 1] += missed
        hits = float(np.mean([len(set(found[qi].tolist()) & set(targets[qi].tolist())) / 100 for qi in range(len(targets))]))
        return table, place, hits

    def summary(table, hits):
        return {'recall_at_100': hits,
                'miss_rate': {str(c): float(table[:, c - 1, :, 1].sum() / table[:, c - 1, :, 0].sum()) for c in (1, 2, 3)}}

    decile = position * 10 // n
    found_at = {ef: search(ef)}
    per_query, by_position, recall = score(found_at[ef])
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'per-query.npy', per_query)
    np.save(output / 'per-query-position.npy', by_position)
    np.save(output / f'labels-ef{ef}.npy', found_at[ef].astype(np.int32))
    widths = {}
    for width in efs:
        if width == ef:
            table, place, hits = per_query, by_position, recall
        else:
            found_at[width] = search(width)
            table, place, hits = score(found_at[width])
            np.save(output / f'labels-ef{width}.npy', found_at[width].astype(np.int32))
        np.save(output / f'per-query-ef{width}.npy', table)
        np.save(output / f'per-query-position-ef{width}.npy', place)
        widths[str(width)] = summary(table, hits)
    confirmed = {}
    if confirm is not None:
        rows, targets = confirm
        for width in sorted({ef, *efs}):
            found = search(width, rows)
            table, place, hits = score(found, targets)
            np.save(output / f'confirm-labels-ef{width}.npy', found.astype(np.int32))
            np.save(output / f'confirm-per-query-ef{width}.npy', table)
            np.save(output / f'confirm-per-query-position-ef{width}.npy', place)
            confirmed[str(width)] = summary(table, hits)
    report = {'backend': backend, 'order': order, 'seed': seed, 'ef': ef, 'M': M, 'ef_construction': ef_construction, 'count': int(n),
              'metric': metric,
              'build_seconds': build_seconds, 'recall_at_100': recall,
              'cohorts': {str(c): {'documents': int((cohort == c).sum()),
                                   'mean_insertion_position': float(position[cohort == c].mean() / n),
                                   'indegree_mean': float(indegree[cohort == c].mean()),
                                   'indegree_median': float(np.median(indegree[cohort == c])),
                                   'indegree_below_5_share': float((indegree[cohort == c] < 5).mean()),
                                   'indegree_zero_share': float((indegree[cohort == c] == 0).mean()),
                                   'outdegree_mean': float(outdegree[cohort == c].mean()),
                                   'miss_rate': float(per_query[:, c - 1, :, 1].sum() / per_query[:, c - 1, :, 0].sum())}
                          for c in (1, 2, 3)},
              # In-degree by insertion decile, whatever the cohort.
              'indegree_by_insertion_decile': [float(indegree[(position * 10 // n) == d].mean()) for d in range(10)]}
    if widths:
        report['widths'] = widths
    if confirmed:
        report['confirm'] = {'queries': int(len(confirm[0])), 'widths': confirmed}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def _ratio(total, target, reference):
    """Observed misses of the target cohorts over the misses expected at the reference cohort's
    miss rate in each exact-rank band. `total` is [cohort, band, (n, missed)]; cohorts are 1-based."""
    target = [c - 1 for c in ((target,) if isinstance(target, int) else target)]
    observed = total[target, :, 1].sum()
    expected = sum(total[target, b, 0].sum() * total[reference - 1, b, 1] / max(1, total[reference - 1, b, 0])
                   for b in range(len(BANDS)))
    return float(observed / expected) if expected else float('nan')   # undefined without reference misses


def _draws(queries, draws, seed):
    """The same query resamples for every build, so that builds can be compared within a draw."""
    return np.random.default_rng(seed).integers(0, queries, (draws, queries))


def ratio(per_query, target, reference, *, draws=5000, seed=20261120):
    """Rank-standardized miss ratio of cohort(s) `target` vs cohort `reference`, with a query bootstrap."""
    boot = [_ratio(per_query[rows].sum(axis=0), target, reference) for rows in _draws(len(per_query), draws, seed)]
    return {'ratio': float(_ratio(per_query.sum(axis=0), target, reference)),
            'interval_95': np.quantile(boot, [.025, .975]).tolist()}


def _table(folder, ef=None):
    """Per-query miss table of one build, at its primary width or at one of its further widths."""
    return np.load(Path(folder) / ('per-query.npy' if ef is None else f'per-query-ef{ef}.npy'))


def summarize(root, ef=None):
    out = []
    for path in sorted(Path(root).glob('*/report.json')):
        report = json.loads(path.read_text())
        report.setdefault('backend', 'faiss_hnsw')          # the first pilot's reports predate the field
        per_query = _table(path.parent, ef)
        if ef is not None:
            report['ef'] = ef
            report['recall_at_100'] = report['widths'][str(ef)]['recall_at_100']
        total = per_query.sum(axis=0)
        # Numerators and denominators behind every ratio: [cohort][band] = exact neighbours, missed.
        report['counts'] = {str(c): {f'{lo}-{hi}': {'exact_neighbours': int(total[c - 1, b, 0]), 'missed': int(total[c - 1, b, 1])}
                                     for b, (lo, hi) in enumerate(BANDS)} for c in (1, 2, 3)}
        report['miss_ratio_cohort3_vs_cohort1'] = ratio(per_query, 3, 1)
        report['miss_ratio_cohort2_vs_cohort1'] = ratio(per_query, 2, 1)
        report['miss_ratio_later_vs_cohort1'] = ratio(per_query, (2, 3), 1)
        out.append(report)
    return out


def contrast(root, *, ef=None, draws=5000, seed=20261120):
    """Paired comparison of insertion orders on the same queries.

    For each seed and for the mean over seeds: the miss ratio in each order, and the as-arrived
    ratio minus the reversed ratio. Every build is resampled with the same query draws, so the
    interval of the difference is paired.
    """
    builds = {}
    for path in sorted(Path(root).glob('*/report.json')):
        report = json.loads(path.read_text())
        builds[(report['order'], report['seed'])] = _table(path.parent, ef)
    seeds = sorted({seed_ for _, seed_ in builds})
    rows = _draws(len(next(iter(builds.values()))), draws, seed)
    out = {}
    for name, target in (('cohort3_vs_cohort1', 3), ('cohort2_vs_cohort1', 2), ('later_vs_cohort1', (2, 3))):
        point = {key: _ratio(per_query.sum(axis=0), target, 1) for key, per_query in builds.items()}
        boot = {key: np.array([_ratio(per_query[r].sum(axis=0), target, 1) for r in rows]) for key, per_query in builds.items()}

        def item(value, sample):
            return {'value': float(value), 'interval_95': np.quantile(sample, [.025, .975]).tolist()}
        block = {'by_seed': {}, 'seed_mean': {}}
        for s in seeds:
            block['by_seed'][str(s)] = {
                'as_arrived_minus_reversed': item(point[('appended', s)] - point[('reversed', s)],
                                                  boot[('appended', s)] - boot[('reversed', s)]),
                'as_arrived_minus_shuffled': item(point[('appended', s)] - point[('shuffled', s)],
                                                  boot[('appended', s)] - boot[('shuffled', s)])}
        mean = {order: (np.mean([point[(order, s)] for s in seeds]), np.mean([boot[(order, s)] for s in seeds], axis=0))
                for order in ('appended', 'shuffled', 'reversed')}
        for order, (value, sample) in mean.items():
            block['seed_mean'][order] = item(value, sample)
        block['seed_mean']['as_arrived_minus_reversed'] = item(mean['appended'][0] - mean['reversed'][0],
                                                               mean['appended'][1] - mean['reversed'][1])
        block['seed_mean']['as_arrived_minus_shuffled'] = item(mean['appended'][0] - mean['shuffled'][0],
                                                               mean['appended'][1] - mean['shuffled'][1])
        block['seed_mean']['shuffled_minus_reversed'] = item(mean['shuffled'][0] - mean['reversed'][0],
                                                             mean['shuffled'][1] - mean['reversed'][1])
        out[name] = block
    return out


def position_curve(folder, *, order='shuffled', ef=None, draws=5000, seed=20261120):
    """Misses by insertion decile for the builds of one order: the miss rate of each decile and its
    rank-standardized ratio to the first decile, as the mean over seeds with a query bootstrap.

    In a shuffled build a document's decile is independent of the document, so the curve shows the
    effect of position alone.
    """
    name = 'per-query-position.npy' if ef is None else f'per-query-position-ef{ef}.npy'
    tables = [np.load(path.parent / name) for path in sorted(Path(folder).glob('*/report.json'))
              if json.loads(path.read_text())['order'] == order]
    if not tables:
        raise ValueError(f'no {order} build with position tables in {folder}')
    totals = [table.sum(axis=0) for table in tables]
    boot = np.empty((draws, 9))
    for i, rows in enumerate(_draws(len(tables[0]), draws, seed)):
        sums = [table[rows].sum(axis=0) for table in tables]
        boot[i] = [np.mean([_ratio(total, d, 1) for total in sums]) for d in range(2, 11)]
    return {'order': order, 'builds': len(tables),
            'miss_rate_by_insertion_decile': [float(np.mean([total[d, :, 1].sum() / total[d, :, 0].sum() for total in totals]))
                                              for d in range(10)],
            'ratio_to_first_decile': [{'decile': d, 'value': float(np.mean([_ratio(total, d, 1) for total in totals])),
                                       'interval_95': np.quantile(boot[:, d - 2], [.025, .975]).tolist()}
                                      for d in range(2, 11)]}


def robustness(sample, roots, output):
    """Aggregates of builds searched at several widths: {root name: {backend: {ef: ...}}}.

    For each width: recall, the seed-mean miss ratios of each order with paired contrasts, and the
    missed neighbours by cohort. Counts and ratios only.
    """
    out = {'kind': 'EXPLORATORY insertion-order pilot: graph settings and search widths (aggregates only)',
           'sample': json.loads((Path(sample) / 'sample.json').read_text()),
           'bootstrap': {'draws': 5000, 'seed': 20261120, 'unit': 'query'}, 'settings': {}}
    for root in map(Path, roots):
        reports = [json.loads(path.read_text()) for path in sorted(root.glob('*/*/report.json'))]   # root/backend/build
        widths = sorted({int(w) for r in reports for w in r['widths']})
        block = {'M': reports[0]['M'], 'ef_construction': reports[0]['ef_construction'],
                 'metric': reports[0].get('metric', 'ip'), 'backends': {}}
        if any((r['M'], r['ef_construction']) != (block['M'], block['ef_construction']) for r in reports):
            raise ValueError(f'mixed graph settings in {root}')
        for backend in sorted({r['backend'] for r in reports}):
            mine = [r for r in reports if r['backend'] == backend]
            folder = root / backend
            per_width = {}
            for ef in widths:
                builds = summarize(folder, ef)
                per_width[str(ef)] = {
                    'recall_at_100': {order: [b['recall_at_100'] for b in builds if b['order'] == order]
                                      for order in ('appended', 'shuffled', 'reversed')},
                    'missed_by_cohort': {f"{b['order']}-{b['seed']}": {c: sum(v['missed'] for v in b['counts'][c].values())
                                                                     for c in '123'} for b in builds},
                    'contrasts': contrast(folder, ef=ef)}
                if all((path.parent / f'per-query-position-ef{ef}.npy').exists() for path in folder.glob('*/report.json')):
                    per_width[str(ef)]['position_curve_shuffled'] = position_curve(folder, order='shuffled', ef=ef)
            block['backends'][backend] = {
                'builds': len(mine), 'build_minutes_median': float(np.median([r['build_seconds'] for r in mine]) / 60),
                'indegree_by_insertion_decile': {f"{r['order']}-{r['seed']}": r['indegree_by_insertion_decile'] for r in mine},
                'widths': per_width}
        out['settings'][root.name] = block
    Path(output).write_text(json.dumps(out, indent=2) + '\n')
    return out


def aggregate(sample, roots, output):
    """One aggregate file for the pilot: sample identity, per-build counts and ratios, paired contrasts.

    It holds counts and ratios only, no identifiers, vectors or rankings.
    """
    out = {'kind': 'EXPLORATORY insertion-order pilot on a sample (aggregates only)',
           'sample': json.loads((Path(sample) / 'sample.json').read_text()),
           'bands': [f'{lo}-{hi}' for lo, hi in BANDS], 'bootstrap': {'draws': 5000, 'seed': 20261120, 'unit': 'query'},
           'builds': [], 'contrasts': {}}
    for root in roots:
        builds = summarize(root)
        out['builds'].extend(builds)
        out['contrasts'][builds[0]['backend']] = contrast(root)
    Path(output).write_text(json.dumps(out, indent=2) + '\n')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('dense', 'inputs', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--count', type=int, default=300_000)
    p = sub.add_parser('prepare-full', help='all of snapshot 3 (draw 0) or a stratified draw of snapshot-1 size')
    for name in ('dense', 'inputs', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--draw', type=int, default=0)
    p.add_argument('--count', type=int)
    p = sub.add_parser('collect', help='gather returned full-scale job outputs of one dataset into one folder')
    p.add_argument('jobs', type=Path)
    p.add_argument('dataset', choices=('full', 'draw1', 'draw2', 'draw3', 'full-m32'))
    p.add_argument('target', type=Path)
    p = sub.add_parser('compare-heldout', help='a chronological full build against the held-out incremental run')
    for name in ('sample', 'build', 'baseline'):
        p.add_argument(name, type=Path)
    p = sub.add_parser('prepare-public', help='a public .fvecs benchmark with random arrival groups')
    for name in ('base', 'queries', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--metric', choices=('cosine', 'l2'), required=True)
    p.add_argument('--count', type=int, default=300_000)
    p = sub.add_parser('run')
    p.add_argument('sample', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--order', choices=('appended', 'reversed', 'shuffled'), required=True)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--backend', choices=('faiss_hnsw', 'epzero'), default='faiss_hnsw')
    p.add_argument('--M', type=int, default=16)
    p.add_argument('--ef-construction', type=int, default=100)
    p.add_argument('--efs', type=int, nargs='*', default=(), help='further search widths for the same build')
    p = sub.add_parser('summary')
    p.add_argument('root', type=Path)
    p = sub.add_parser('aggregate')
    p.add_argument('sample', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('roots', type=Path, nargs='+')
    p = sub.add_parser('robustness')
    p.add_argument('sample', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('roots', type=Path, nargs='+')
    a = parser.parse_args()
    if a.command == 'prepare':
        r = prepare(a.dense, a.inputs, a.output, count=a.count)
    elif a.command == 'prepare-full':
        r = prepare_full(a.dense, a.inputs, a.output, draw=a.draw, count=a.count)
    elif a.command == 'collect':
        r = collect(a.jobs, a.dataset, a.target)
    elif a.command == 'compare-heldout':
        r = compare_heldout(a.sample, a.build, a.baseline)
    elif a.command == 'prepare-public':
        r = prepare_public(a.base, a.queries, a.output, metric=a.metric, count=a.count)
    elif a.command == 'run':
        r = run(a.sample, a.output, order=a.order, seed=a.seed, backend=a.backend, M=a.M,
                ef_construction=a.ef_construction, efs=tuple(a.efs))
        r = {k: r[k] for k in ('order', 'seed', 'build_seconds', 'recall_at_100', 'cohorts', 'widths') if k in r}
    elif a.command == 'aggregate':
        r = aggregate(a.sample, a.roots, a.output)['contrasts']
    elif a.command == 'robustness':
        r = {name: {b: list(v['widths']) for b, v in block['backends'].items()}
             for name, block in robustness(a.sample, a.roots, a.output)['settings'].items()}
    else:
        r = summarize(a.root)
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
