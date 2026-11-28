"""A3: which links pruning removes (docs/a3-mechanism-derivation.md, section 6; design recorded before the build).

    python -m research.longeval.a3_links run SAMPLE OUT [--seed 11] [--rate 0.02]
    python -m research.longeval.a3_links analyze OUT SUMMARY.json

`run` builds M1's configuration one insertion at a time and, after each insertion, reads the layer-0 lists of the
new node and of every node it chose. Comparing them with their previous state gives each link's creation step and
the step it is pruned. A sampled share of pruning events is written with every candidate: owner, candidate, age in
insertion steps, squared distance to the owner, removed or kept. `analyze` scores the five predictions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .insertion_pilot import insertion_order

EVENT_SEED = 2026


def run(sample, output, *, seed=11, rate=0.02, M=16, ef_construction=100, limit=None):
    import epzero
    sample, output = Path(sample), Path(output)
    vectors = np.load(sample / "vectors.npy", mmap_mode="r")
    cohort = np.load(sample / "cohort.npy")
    metric = json.loads((sample / "sample.json").read_text()).get("metric", "ip")
    if metric != "l2":
        raise ValueError("the distance ranks below assume L2")
    sequence = insertion_order(np.load(sample / "lexical-rank.npy"), cohort, "shuffled", seed)
    if limit:
        sequence = sequence[:limit]
    n, dim = len(sequence), vectors.shape[1]
    index = epzero.Index(dim=dim, metric=metric, max_elements=n, M=M, ef_construction=ef_construction, seed=seed,
                         quantization="none", keep_vectors=True, nav_dim=0, fill_links=False, fast_build=False,
                         half_vectors=False)
    rng = np.random.default_rng(EVENT_SEED)
    born = {}                                  # owner -> {target: creation step}
    rows = []                                  # (event, step, owner_position, candidate_position, age, distance, removed)
    events = recorded = refused = untracked = 0
    removed_counts = np.zeros(8, dtype=np.int64)     # links removed per pruning event (7 = 7 or more)
    position = np.full(vectors.shape[0], -1, dtype=np.int64)
    started = time.perf_counter()
    for p, label in enumerate(sequence):
        label = int(label)
        index.add(np.ascontiguousarray(vectors[label:label + 1]), labels=np.array([label], dtype=np.int64), threads=1)
        position[label] = p
        mine = [int(y) for y in index.neighbors(label, 0)]
        born[label] = {y: p for y in mine}
        for x in mine:
            now = set(int(y) for y in index.neighbors(x, 0))
            before = born[x]
            untracked += sum(1 for y in now if y not in before and y != label)   # must stay 0: only chosen lists change
            gone = [y for y in before if y not in now]
            accepted = label in now
            if not accepted:
                refused += 1
            if gone or not accepted:
                events += 1
                removed_counts[min(len(gone) + (not accepted), 7)] += 1
                if rng.random() < rate:
                    candidates = list(before) + [label]
                    v = np.asarray(vectors[candidates], dtype=np.float64)
                    d = ((v - np.asarray(vectors[x], dtype=np.float64)) ** 2).sum(axis=1)
                    for c, dist in zip(candidates, d):
                        age = p - before.get(c, p)
                        rows.append((recorded, p, position[x], position[c], age, float(dist), int(c not in now)))
                    recorded += 1
            for y in gone:
                del before[y]
            if accepted:
                before[label] = p
        if (p + 1) % 50_000 == 0:
            print(json.dumps({"inserted": p + 1, "events": events, "recorded": recorded,
                              "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    mismatched = sum(1 for node, links in born.items() if set(links) != set(int(y) for y in index.neighbors(node, 0)))
    output.mkdir(parents=True, exist_ok=False)
    table = np.array(rows, dtype=np.float64)
    np.save(output / "events.npy", table)
    report = {"design": "docs/a3-mechanism-derivation.md section 6", "count": int(n), "M": M,
              "ef_construction": ef_construction, "seed": seed, "order": "shuffled", "epzero": epzero.__version__,
              "event_rate": rate, "event_seed": EVENT_SEED, "pruning_events": int(events), "recorded_events": int(recorded),
              "refused_reverse_links": int(refused), "removed_per_event": removed_counts.tolist(),
              "untracked_links": int(untracked), "lists_differing_from_tracking_at_end": int(mismatched),
              "seconds": time.perf_counter() - started}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _ranks(values):
    """Ranks scaled to [0, 1] (0 smallest); ties share their mean rank."""
    order = np.argsort(values, kind="stable")
    r = np.empty(len(values))
    r[order] = np.arange(len(values))
    for v in np.unique(values):
        same = values == v
        if same.sum() > 1:
            r[same] = r[same].mean()
    return r / max(len(values) - 1, 1)


def logistic(X, y, iterations=50):
    """Maximum-likelihood logistic regression with an intercept, by Newton's method."""
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-X @ beta))
        grad = X.T @ (y - p)
        hess = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def analyze(output):
    output = Path(output)
    report = json.loads((output / "report.json").read_text())
    t = np.load(output / "events.npy")
    event, age, dist, removed = t[:, 0].astype(np.int64), t[:, 4], t[:, 5], t[:, 6]
    age_rank, dist_rank, rho = np.empty(len(t)), np.empty(len(t)), []
    for e in np.unique(event):
        rows = np.flatnonzero(event == e)
        age_rank[rows], dist_rank[rows] = _ranks(age[rows]), _ranks(dist[rows])
        a, d = age_rank[rows], dist_rank[rows]
        if a.std() > 0 and d.std() > 0:
            rho.append(float(np.corrcoef(a, d)[0, 1]))
    gone = removed == 1
    age_only, dist_only, both = (logistic(age_rank[:, None], removed), logistic(dist_rank[:, None], removed),
                                 logistic(np.column_stack([age_rank, dist_rank]), removed))
    counts = np.array(report["removed_per_event"])
    one = float(counts[1] / counts.sum())
    out = {"kind": "EXPLORATORY A3 link-history build (docs/a3-mechanism-derivation.md section 6), aggregates only",
           "build": report, "recorded_candidates": int(len(t)),
           "removed_mean_age_rank": float(age_rank[gone].mean()), "kept_mean_age_rank": float(age_rank[~gone].mean()),
           "removed_mean_distance_rank": float(dist_rank[gone].mean()), "kept_mean_distance_rank": float(dist_rank[~gone].mean()),
           "mean_within_event_spearman_age_distance": float(np.mean(rho)),
           "logistic": {"age_only": age_only.tolist(), "distance_only": dist_only.tolist(), "both": both.tolist()},
           "share_of_events_removing_exactly_one": one,
           "removed_share_by_age_rank_fifth": [float(removed[(age_rank >= q / 5) & (age_rank < (q + 1) / 5 + (q == 4))].mean())
                                               for q in range(5)],
           "removed_share_by_distance_rank_fifth": [float(removed[(dist_rank >= q / 5) & (dist_rank < (q + 1) / 5 + (q == 4))].mean())
                                                    for q in range(5)]}
    out["predictions"] = {
        "1_removed_age_rank_above_0.6": bool(out["removed_mean_age_rank"] > 0.6),
        "2_removed_distance_rank_above_0.6": bool(out["removed_mean_distance_rank"] > 0.6),
        "3_spearman_age_distance_above_0.1": bool(out["mean_within_event_spearman_age_distance"] > 0.1),
        "4_age_coefficient_halves_with_distance": bool(abs(both[1]) < 0.5 * abs(age_only[1])),
        "5_at_least_80pct_remove_exactly_one": bool(one >= 0.8)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("sample", type=Path)
    r.add_argument("output", type=Path)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--rate", type=float, default=0.02)
    r.add_argument("--limit", type=int)
    a = sub.add_parser("analyze")
    a.add_argument("output", type=Path)
    a.add_argument("summary", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        print(json.dumps(run(args.sample, args.output, seed=args.seed, rate=args.rate, limit=args.limit), indent=2))
    else:
        out = analyze(args.output)
        args.summary.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({k: out[k] for k in ("predictions", "removed_mean_age_rank", "removed_mean_distance_rank",
                                              "mean_within_event_spearman_age_distance", "share_of_events_removing_exactly_one",
                                              "logistic")}, indent=2))


if __name__ == "__main__":
    main()
