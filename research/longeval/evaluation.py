"""Independent pilot scorer; explicit cohorts, linear gains and deterministic ties."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .contracts import Table, token


def rank(pairs):
    return sorted(pairs, key=lambda x: (x[1], x[0]), reverse=True)


def validate_run(run, query_ids, live_ids, k):
    if set(run) != set(query_ids):
        raise ValueError("run query population mismatch")
    live = set(live_ids)
    for pairs in run.values():
        ids = [d for d, _ in pairs]
        if len(pairs) != min(k, len(live)) or len(ids) != len(set(ids)):
            raise ValueError("short run or duplicate results")
        if set(ids) - live or not all(math.isfinite(s) for _, s in pairs):
            raise ValueError("invalid output ID or nonfinite score")


def exact(table: Table, queries: Table, k: int, metric: str, block_size: int = 4096):
    """Exhaustive float32 inputs, float64 accumulation; one query and one block at a time.

    L2 uses direct subtraction, avoiding norm-expansion cancellation. Scores are higher-is-better:
    -squared-L2, inner product, or dot-1 for cosine (matching Epzero's negative distance).
    """
    if k <= 0 or block_size <= 0 or metric not in ("l2", "ip", "cosine"):
        raise ValueError("invalid exact-search configuration")
    result = {}
    for qid, query in zip(queries.ids, queries.vectors):
        best = []
        q = query.astype(np.float64)
        for start in range(0, len(table.ids), block_size):
            block = table.vectors[start:start + block_size].astype(np.float64)
            if metric == "l2":
                diff = block - q
                scores = -np.einsum("ij,ij->i", diff, diff)
            else:
                scores = block @ q - (1 if metric == "cosine" else 0)
            best = rank(best + list(zip(table.ids[start:start + block_size], map(float, scores))))[:k]
        result[qid] = best
    return result


def evaluate(run, qrels, query_ids, *, population, ndcg_k=10, recall_k=100):
    if population not in ("explicit", "positive", "judged"):
        raise ValueError("unknown evaluation population")
    if ndcg_k <= 0 or recall_k <= 0 or len(query_ids) != len(set(query_ids)):
        raise ValueError("invalid metric depth or duplicate query IDs")
    cohort = [q for q in query_ids if population == "explicit" or
              (population == "judged" and q in qrels) or
              (population == "positive" and any(r > 0 for r in qrels.get(q, {}).values()))]
    rows = {}
    for q in cohort:
        judgments = qrels.get(q, {})
        docs = [d for d, _ in rank(run.get(q, []))]
        gains = [max(0, judgments.get(d, 0)) for d in docs[:ndcg_k]]
        ideal = sorted((max(0, r) for r in judgments.values()), reverse=True)[:ndcg_k]
        discount = lambda values: sum(g / math.log2(i + 2) for i, g in enumerate(values))
        idcg = discount(ideal)
        relevant = {d for d, r in judgments.items() if r > 0}
        rows[q] = {f"nDCG@{ndcg_k}": discount(gains) / idcg if idcg else 0.0,
                   f"R@{recall_k}": len(set(docs[:recall_k]) & relevant) / len(relevant) if relevant else 0.0,
                   f"judged@{ndcg_k}": sum(d in judgments and judgments[d] >= 0 for d in docs[:ndcg_k]) / ndcg_k}
    means = {name: sum(row[name] for row in rows.values()) / len(rows)
             for name in next(iter(rows.values()), {})}
    return {"population": population, "count": len(cohort), "query_ids": cohort,
            "per_query": rows, "mean": means}


def ann_recall(run, reference):
    if set(run) != set(reference):
        raise ValueError("ANN recall populations differ")
    return {q: len({d for d, _ in run[q]} & {d for d, _ in ref}) / len(ref) if ref else 0.0
            for q, ref in reference.items()}


def tie_aware_recall(run, reference, table, queries, metric, atol=1e-5):
    """Credit boundary-tied substitutions, while still requiring every strictly better result.

    Uses canonical scores, not potentially approximate backend scores. atol is recorded as a
    diagnostic tolerance, not used to change the relevance ranking.
    """
    positions = {key: i for i, key in enumerate(table.ids)}
    out = {}
    for qid, vector in zip(queries.ids, queries.vectors):
        ref = reference[qid]
        cutoff = ref[-1][1]
        strict = {d for d, s in ref if s > cutoff + atol}
        candidates = [d for d, _ in run[qid]]
        v = table.vectors[[positions[d] for d in candidates]].astype(np.float64)
        q = vector.astype(np.float64)
        scores = -np.sum((v - q) ** 2, axis=1) if metric == "l2" else v @ q - (metric == "cosine")
        boundary = sum(abs(float(s) - cutoff) <= atol for s in scores)
        out[qid] = (len(strict & set(candidates)) + min(len(ref) - len(strict), boundary)) / len(ref)
    return out


def write_run(path: Path, run):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for qid in sorted(run):
            for i, (doc, score) in enumerate(rank(run[qid]), 1):
                token(qid)
                token(doc)
                if not math.isfinite(score):
                    raise ValueError("nonfinite run score")
                stream.write(f"{qid} Q0 {doc} {i} {score:.17g} longeval\n")


def read_run(path: Path):
    run = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        q, _, doc, position, score, _ = line.split()
        pairs = run.setdefault(q, [])
        if int(position) != len(pairs) + 1 or not math.isfinite(float(score)):
            raise ValueError("invalid run rank or score")
        pairs.append((doc, float(score)))
    return run
