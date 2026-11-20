"""Bounded exhaustive scoring for a memory-mapped float32 document matrix.

This is an independent reference implementation, not an ANN index. Every live vector is scored.
Queries are tiled within each document block so the corpus is read once, with top-k retained only.
"""
from __future__ import annotations

import mmap

import numpy as np

from .evaluation import rank


def _topk(ids, scores, k):
    """Retain all candidates above the kth score and resolve its exact ties by external ID."""
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite exhaustive scores")
    if len(scores) <= k:
        return rank([(ids[i], float(score)) for i, score in enumerate(scores)])
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > cutoff)
    ties = np.flatnonzero(scores == cutoff)
    remaining = k - len(above)
    chosen = sorted(ties, key=lambda i: ids[int(i)], reverse=True)[:remaining]
    return rank([(ids[int(i)], float(scores[int(i)])) for i in [*above, *chosen]])


def exact_mapped(document_ids, vectors, query_ids, query_vectors, *, k=100, metric="ip",
                 block_rows=2048, query_batch=32, progress=None):
    """Float64 accumulation over original float32 inputs, ordered by descending score then ID.

    For IP/cosine, a query tile uses matrix multiplication. L2 retains direct subtraction to avoid
    norm-expansion cancellation. Cosine requires pre-normalized inputs; it never normalizes data.
    The caller must establish input identity and representation validity with the ingestion audit.
    Matrix products can differ by rounding from the scalar-query oracle; compare with tolerances.
    """
    if (type(k) is not int or k <= 0 or type(block_rows) is not int or block_rows <= 0
            or type(query_batch) is not int or query_batch <= 0 or metric not in {"ip", "cosine", "l2"}):
        raise ValueError("invalid exhaustive scoring configuration")
    if (vectors.ndim != 2 or query_vectors.ndim != 2 or vectors.shape[1] != query_vectors.shape[1]
            or len(document_ids) != len(vectors) or len(query_ids) != len(query_vectors)
            or vectors.dtype != np.dtype("<f4") or query_vectors.dtype != np.dtype("<f4")):
        raise ValueError("exhaustive input dtype/shape mismatch")
    if len(set(document_ids)) != len(document_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("duplicate exhaustive input IDs")
    if not np.isfinite(query_vectors).all():
        raise ValueError("nonfinite query vectors")
    if metric == "cosine" and not np.allclose(np.linalg.norm(query_vectors.astype(np.float64), axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("cosine query vectors are not unit normalized")
    result = {qid: [] for qid in query_ids}
    queries = query_vectors.astype(np.float64)
    for start in range(0, len(document_ids), block_rows):
        end = min(len(document_ids), start + block_rows)
        block = np.array(vectors[start:end], dtype=np.float64, copy=True)
        if not np.isfinite(block).all():
            raise ValueError("nonfinite document vectors")
        if metric == "cosine" and not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-5, rtol=0):
            raise ValueError("cosine document vectors are not unit normalized")
        ids = document_ids[start:end]
        for qstart in range(0, len(query_ids), query_batch):
            qend = min(len(query_ids), qstart + query_batch)
            if metric in {"ip", "cosine"}:
                scores = block @ queries[qstart:qend].T
                if metric == "cosine":
                    scores -= 1
            else:
                scores = np.empty((len(block), qend - qstart), dtype=np.float64)
                for offset, query in enumerate(queries[qstart:qend]):
                    difference = block - query
                    scores[:, offset] = -np.einsum("ij,ij->i", difference, difference)
            for offset, qid in enumerate(query_ids[qstart:qend]):
                result[qid] = rank(result[qid] + _topk(ids, scores[:, offset], k))[:k]
        del block
        if isinstance(vectors, np.memmap) and hasattr(vectors._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
            vectors._mmap.madvise(mmap.MADV_DONTNEED)
        if progress:
            progress({"rows_scored": end, "total_rows": len(document_ids)})
    return result
