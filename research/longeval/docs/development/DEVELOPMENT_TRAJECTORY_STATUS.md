# Development trajectory status — 2026-11-16

The fixed-vector and fixed-text audits pass for all three full snapshots. GPU training-query
encoding and full-corpus exact precision sensitivity are complete. Float32 remains the declared
development candidate because it agrees more closely with the archived document sample; BF16
is a separate sensitivity condition. Neither reproduces the historical runtime exactly.

## Development relevance and evaluator parity

Snapshot 1 exact retrieval was scored on the 50 development training queries only. Neither
calibration-query relevance nor test judgments entered this step. Each per-query nDCG@10 and
Recall@100 value agrees with independent `pytrec_eval` 0.5.11 within 1e-12.

| Training judgments | Precision | nDCG@10 | Recall@100 | Judged@10 |
|---|---|---:|---:|---:|
| DCTR | BF16 | 0.300795 | 0.676258 | 0.534 |
| DCTR | Float32 | 0.297205 | 0.674591 | 0.532 |
| Raw | BF16 | 0.323026 | 0.683910 | 0.218 |
| Raw | Float32 | 0.319889 | 0.682244 | 0.216 |

These are precision diagnostics on an explicit development population. They establish neither
statistical significance nor official-task evaluator/population certification. Precision was not
selected by maximizing these relevance values. Evidence: `build/longeval-development-relevance-01/report.json`.

## New full-data runner

`trajectory.py` runs one HNSW backend, construction mode and seed per process. Incremental
construction persists one native index across snapshots. Replay rebuilds the same chronological
insertion sequence; shuffled rebuild changes its order using an explicit seed. Every stage joins
by document ID, validates vector bytes, inserts bounded batches and closes mapped input pages
before search. At most one native index is retained. Epzero reserves final capacity; FAISS uses
native growth. This difference is recorded rather than treated as matched memory cost.

The development configuration uses M=16, efConstruction=100, frozen ef=200 and depth 100.
Calibration searches ef in [100, 200, 400, 800] on the separate 50 calibration queries, targeting
0.95 ANN recall. Target misses are explicitly recorded; the fallback is not labeled a successful
matched-recall point. Only the 50 development queries are reported as evaluation runs.

Ten targeted tests pass locally: both native backends, three construction modes, one-index
lifetime, replay equivalence, bounded insertion, corruption rejection, memory hashing and
development evaluator parity. FAISS's serialized efSearch is standardized before fingerprinting,
so a prior calibration does not create a false construction-state difference. These tests use
small fixtures and do not establish full-corpus resource feasibility.

The first PACE run is Epzero incremental, seed 11, across all 869,902 / 1,322,720 / 1,661,900
documents. It requests an exclusive Gold 6226 node, one computation thread, 64 GB RAM, 160 GB
node-local temporary disk and eight hours. All new large files stay in that temporary directory.
Only source, small input evidence, reports and runs persist in the shared checkout. This is an
initial resource and correctness run; its instrumentation is not validated publication timing.

## Remaining research work

No full-data ANN trajectory has completed yet. After inspecting the initial run, execute replay
and shuffled controls, the independent FAISS backend and multiple seeds. Establish resource
budgets, matched operating regimes, instrumentation overhead and reconstruction limits. Extend
IVF controls and required analyses, freeze primary configuration, practical-effect margins and
the inferential plan before final test comparisons. Novelty assessment, complete experiments,
paper writing and peer review remain outstanding.
