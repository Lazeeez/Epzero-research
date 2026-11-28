"""Screen A1: the arXiv stream in the order study's sample format, and the repaired arm (docs/screen-a1-arxiv-stream.md).

    python -m research.arxiv_stream.stream sample VECTORS PREPARED SAMPLE
    python -m research.longeval.insertion_pilot run SAMPLE OUT --order appended|reversed|shuffled --backend epzero|faiss_hnsw --seed S --efs 100 200 400 800
    python -m research.arxiv_stream.stream repaired SAMPLE OUT --seed S [--threads 8]

`sample` writes what the frozen order runner (`research/longeval/insertion_pilot.py run`, unchanged) reads:
the corpus vectors in submission order, cohorts by thirds of that order, the insertion rank, the
new-content queries as the queries and the uniform queries as its confirmation set (searched at the same
widths, never used to choose anything), each with its exact inner-product top 100; plus each abstract's
month. `repaired` is the one arm that runner lacks: Epzero in the appended order with `extra_backlinks` 8,
refined at a beam of 100 after each calendar month's insertions, scored exactly as the runner scores.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from research.longeval.accumulation import exact_prefix
from research.longeval.insertion_pilot import BANDS

REPAIR_LINKS, REPAIR_EF = 8, 100
WIDTHS = (100, 200, 400, 800)


def _ids(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _months(prepared):
    with gzip.open(Path(prepared) / "corpus.jsonl.gz", "rt", encoding="utf-8") as f:
        return np.array([int(json.loads(line)["created"][:7].replace("-", "")) for line in f], dtype=np.int32)


def cohorts(n):
    """1, 2, 3 for the oldest, middle and newest thirds of the insertion order, by count."""
    return (1 + np.arange(n, dtype=np.int64) * 3 // n).astype(np.int8)


def sample(vectors, prepared, output):
    vectors, prepared, output = Path(vectors), Path(prepared), Path(output)
    report = json.loads((vectors / "report.json").read_text())
    if report.get("status") != "completed":
        raise ValueError("the encoding did not complete")
    with gzip.open(prepared / "corpus.jsonl.gz", "rt", encoding="utf-8") as f:
        prepared_ids = [json.loads(line)["id"] for line in f]
    if _ids(vectors / "corpus-ids.json.gz") != prepared_ids:
        raise ValueError("the vectors are not in the prepared corpus's order")
    corpus = np.load(vectors / "corpus.npy", mmap_mode="r")
    new, uniform = np.load(vectors / "queries-new.npy"), np.load(vectors / "queries-uniform.npy")
    n = len(corpus)
    started = time.perf_counter()
    exact_new = exact_prefix(corpus, new, np.arange(n), "ip").astype(np.int32)
    exact_uniform = exact_prefix(corpus, uniform, np.arange(n), "ip").astype(np.int32)
    exact_seconds = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    try:
        os.link(vectors / "corpus.npy", output / "vectors.npy")        # 8.6 GB: a hard link where possible
    except OSError:
        shutil.copyfile(vectors / "corpus.npy", output / "vectors.npy")
    cohort, month = cohorts(n), _months(prepared)
    np.save(output / "cohort.npy", cohort)
    np.save(output / "lexical-rank.npy", np.arange(n, dtype=np.int64))
    np.save(output / "month.npy", month)
    np.save(output / "queries.npy", new)
    np.save(output / "exact-top100.npy", exact_new)
    np.save(output / "confirm-queries.npy", uniform)
    np.save(output / "confirm-exact-top100.npy", exact_uniform)
    digest = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    summary = {"design": "docs/screen-a1-arxiv-stream.md", "dataset": "arxiv-bge-base-submission-order", "metric": "ip",
               "count": int(n), "dimension": int(corpus.shape[1]), "queries": int(len(new)),
               "confirm_queries": int(len(uniform)), "queries_role": "new-content", "confirm_role": "uniform",
               "cohort_sizes": {str(c): int((cohort == c).sum()) for c in (1, 2, 3)},
               "cohort_first_month": {str(c): int(month[cohort == c][0]) for c in (1, 2, 3)},
               "months": int(len(np.unique(month))), "exact_seconds": exact_seconds,
               "encoding_report_sha256": hashlib.sha256((vectors / "report.json").read_bytes()).hexdigest(),
               "derived_sha256": {"cohort": digest(cohort), "month": digest(month), "exact_new": digest(exact_new),
                                  "exact_uniform": digest(exact_uniform)}}
    (output / "sample.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def score(found, targets, cohort, decile):
    """The order runner's per-query tables, unchanged: misses by cohort and by insertion decile within exact-rank
    bands ([query, group, band, (n, missed)]), and mean recall@100."""
    table = np.zeros((len(targets), 3, len(BANDS), 2), dtype=np.int64)
    place = np.zeros((len(targets), 10, len(BANDS), 2), dtype=np.int64)
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


def repaired(sample_dir, output, *, seed, M=16, ef_construction=100, threads=8, widths=WIDTHS, batch=1024):
    import epzero
    sample_dir, output = Path(sample_dir), Path(output)
    vectors = np.load(sample_dir / "vectors.npy", mmap_mode="r")
    cohort, month = np.load(sample_dir / "cohort.npy"), np.load(sample_dir / "month.npy")
    if json.loads((sample_dir / "sample.json").read_text())["metric"] != "ip":
        raise ValueError("A1 is inner product")
    n, dim = vectors.shape
    # The order runner's Epzero configuration (uncompressed), with the repair's extra links.
    index = epzero.Index(dim=dim, metric="ip", max_elements=n, M=M, ef_construction=ef_construction, seed=seed,
                         quantization="none", keep_vectors=True, nav_dim=0, fill_links=False, fast_build=False,
                         half_vectors=False, extra_backlinks=REPAIR_LINKS)
    bounds = np.flatnonzero(np.diff(month)) + 1                # first row of each month after the first
    starts, ends = np.concatenate([[0], bounds]), np.concatenate([bounds, [n]])
    insert_seconds = refine_seconds = 0.0
    refined = 0
    for first, last in zip(starts, ends):
        t = time.perf_counter()
        for start in range(first, last, batch):
            rows = np.arange(start, min(start + batch, last))
            index.add(np.ascontiguousarray(vectors[rows]), labels=rows.astype(np.int64), threads=1)
        insert_seconds += time.perf_counter() - t
        if first > 0:                                         # the first month is the whole graph so far
            t = time.perf_counter()
            refined += index.refine_links(threads=threads, ef=REPAIR_EF, first=int(first))
            refine_seconds += time.perf_counter() - t
    decile = np.arange(n) * 10 // n                           # appended: insertion position = row
    output.mkdir(parents=True, exist_ok=False)
    report = {"backend": "epzero", "order": "appended", "arm": "repaired", "seed": seed, "M": M,
              "ef_construction": ef_construction, "count": int(n), "metric": "ip", "extra_backlinks": REPAIR_LINKS,
              "refine_ef": REPAIR_EF, "refine_threads": threads, "months": int(len(starts)), "refined_nodes": int(refined),
              "insert_seconds": insert_seconds, "refine_seconds": refine_seconds,
              "epzero_version": epzero.__version__, "widths": {}, "confirm": {"widths": {}}}
    for prefix, rows, targets in (("", np.load(sample_dir / "queries.npy"), np.load(sample_dir / "exact-top100.npy")),
                                  ("confirm-", np.load(sample_dir / "confirm-queries.npy"),
                                   np.load(sample_dir / "confirm-exact-top100.npy"))):
        for width in widths:
            labels, _ = index.search(np.ascontiguousarray(rows), k=100, ef=width, rerank=0, rerank_z=0, patience=0,
                                     threads=1)
            found = np.asarray(labels, dtype=np.int64)
            table, place, hits = score(found, targets, cohort, decile)
            np.save(output / f"{prefix}labels-ef{width}.npy", found.astype(np.int32))
            np.save(output / f"{prefix}per-query-ef{width}.npy", table)
            np.save(output / f"{prefix}per-query-position-ef{width}.npy", place)
            summary = {"recall_at_100": hits, "miss_rate": {str(c): float(table[:, c - 1, :, 1].sum() /
                                                                          table[:, c - 1, :, 0].sum()) for c in (1, 2, 3)}}
            (report["confirm"]["widths"] if prefix else report["widths"])[str(width)] = summary
    indegree = np.zeros(n, dtype=np.int64)
    for label in range(n):
        np.add.at(indegree, np.asarray(index.neighbors(label, 0), dtype=np.int64), 1)
    report["cohorts"] = {str(c): {"indegree_mean": float(indegree[cohort == c].mean())} for c in (1, 2, 3)}
    report["indegree_by_insertion_decile"] = [float(indegree[decile == d].mean()) for d in range(10)]
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample")
    for name in ("vectors", "prepared", "output"):
        p.add_argument(name, type=Path)
    p = sub.add_parser("repaired")
    p.add_argument("sample", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--threads", type=int, default=8)
    a = parser.parse_args()
    if a.command == "sample":
        print(json.dumps(sample(a.vectors, a.prepared, a.output), indent=2))
    else:
        r = repaired(a.sample, a.output, seed=a.seed, threads=a.threads)
        print(json.dumps({k: r[k] for k in ("insert_seconds", "refine_seconds", "refined_nodes", "widths")}, indent=2))


if __name__ == "__main__":
    main()
