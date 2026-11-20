# Final evaluation protocol v1 — CANDIDATE for approval, not frozen (2026-11-19)

Scope decided 2026-11-19: an **empirical analysis paper** (CLEF 2027 conference track;
LongEval 2027 working notes if that lab runs). It reports how approximate search,
index history and collection growth affect dense retrieval over three snapshots, plus
four pre-registered method screens that failed at equal work. It makes no new-method
claim. This candidate supersedes FINAL_EVALUATION_PROTOCOL_DRAFT.md once frozen.

Freezing means: this file, the executable configurations and the analysis code are
hashed and timestamped **before any held-out retrieval outcome is produced**. Later
changes create v2 and must state which test outcomes were already visible.

## Research questions

- **RQ1 (history).** Does keeping an index alive while the collection grows change its
  results compared with rebuilding from the same insertion history?
- **RQ2 (approximation over time).** How large is the signed ANN-minus-exact effectiveness
  gap at each snapshot, and does it change from snapshot 1 to snapshot 3?
- **RQ3 (search failures).** How do neighbor recall, failure tails and search work change
  with growth at a fixed versus a calibration-adjusted search setting?
- **RQ4 (repair, development only).** Do query-driven graph repairs beat wider search at
  equal work? Reported from the existing development screens; no held-out runs.

## Data and representation

Original archived document vectors for the three cumulative snapshots, unchanged.
Queries: the 381 test queries, encoded once with Qwen3-Embedding-4B (pinned revision and
prompt), CUDA FP32, TF32 off, batch 1, as for the training queries. Exact and approximate
arms use the same query vectors. Encoding precision is disclosed; it is never chosen on
test outcomes. Training queries keep their 50 calibration / 50 development roles.

## Arms (all single-thread, Gold 6226, construction seeds 11, 29, 47)

| Arm | Settings |
| --- | --- |
| Exact | Float64 exhaustive top-100 (`oracle.py`) |
| Epzero HNSW, incremental | M=16, efC=100; ef 200 frozen and calibration-selected (95% target, grid 100–800) |
| FAISS HNSW, incremental | Same as Epzero |
| Epzero / FAISS HNSW, identical-history rebuild | Same; for RQ1 equality only |
| FAISS IVF-Flat, frozen and refreshed training | nlist 1024, 65,536 training rows, nprobe calibration-selected (95% target) |

Calibration uses the 50 calibration queries' exact neighbor recall at each snapshot.
All seeds are kept and averaged within each query; none is selected.

## Judgments and populations

- **Primary:** GPT-OSS-120B judgments, snapshot-specific, on the fixed panel of test queries
  with at least one positive judgment at all three snapshots (audited: up to 374 queries;
  re-verify identities at freeze). Automated labels; stated as such.
- **Sensitivity, reported separately, never pooled:** raw clicks (80-query common panel),
  DCTR (10-query panel), human labels (29 queries, snapshot 3 only), GPT-OSS-20B.
- Unjudged documents count as non-relevant; judged@10 coverage is reported for every run.

## Outcomes

Primary: nDCG@10 (linear gains, deterministic ties). Secondary: relevant-document R@100,
judged@10, ANN neighbor recall@100, per-query worst-decile and zero-recall counts, distance
computations, native query latency (separate timing jobs, warmup + 3 repeats, median).

## Primary contrasts (one Holm family, three tests)

For ANN arm A in {Epzero HNSW, FAISS HNSW, FAISS IVF-frozen}, at the calibration-selected
setting and primary panel:

    D_A = [nDCG(A, s3) - nDCG(Exact, s3)] - [nDCG(A, s1) - nDCG(Exact, s1)]

Seed-averaged within query, then averaged over queries. Two-sided paired sign-randomization
test (100,000 draws, fixed RNG seed 20261119), Holm-adjusted at 0.05. 95% paired query-
bootstrap percentile intervals (10,000 samples).

**Practical margin: 0.01 absolute nDCG@10.** A gap change is called negligible only if its
whole 95% interval lies within ±0.01. A non-significant result alone is not equivalence.

RQ1 is an equality check, not a test: rankings, scores and state fingerprints of incremental
versus identical-history rebuild must match exactly for every test query, snapshot and seed.
Any mismatch is reported as a finding.

## Reporting rules

Report every arm, seed, snapshot and failed target. Report the exact-search effectiveness
trend separately; it is a property of the representation and labels, not of the index.
Development findings and test findings are labelled separately. No significance claims on
development data. Invalid executions get a documented repair; unfavorable valid outcomes stay.

## Required before freeze

1. User approval of this candidate (primary labels, panel, margin, contrast family).
2. Executable test-query encoding, oracle, trajectory and analysis pipeline, tested on
   synthetic data and on training queries only.
3. Hashes of this file, configurations and code recorded in a freeze receipt.
