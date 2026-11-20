# Exact ranking sensitivity to query encoding precision

All three real snapshots have now been exhaustively scored using the same 100 training queries
encoded in GPU BF16 and GPU float32. Original archived document vectors are unchanged. Query
model files, prompt, tokenization, GPU type and all declared nonprecision encoder settings match;
TF32 is disabled. The query inputs and their 50/50 development/calibration roles were fixed
without judgments. No qrels or test-query contents were used in this diagnostic.

| Snapshot | Documents | Mean top-10 set overlap | Mean top-100 set overlap | Queries with changed top-10 order | Queries with changed top-10 membership |
|---|---:|---:|---:|---:|---:|
| 1 | 869,902 | 99.1% | 99.05% | 51 / 100 | 9 / 100 |
| 2 | 1,322,720 | 99.4% | 99.10% | 54 / 100 | 6 / 100 |
| 3 | 1,661,900 | 98.4% | 98.95% | 55 / 100 | 16 / 100 |

One query's top-ranked document changes in each snapshot. Every query's ordered top-100 list
differs between precisions in every snapshot, while top-100 membership changes for 65, 64 and 67
queries respectively. The minimum per-query set overlap is 90% at depth 10 and 96% at depth 100
in each snapshot. Set overlap and ordered-list equality measure different properties; high
overlap does not imply identical ranking or judged effectiveness.

The paired query vectors themselves have median cosine similarity `0.9997488872694831`.
Thus, near-one vector cosine similarity alone is insufficient evidence of identical retrieval
rankings. These findings identify an experimental control requirement, not an ANN-history effect,
relevance improvement, acceptable-effect threshold or novelty claim.

## Execution and verification

The memory-bounded scorer used float64 accumulation over original float32 documents and query
vectors. It scored both precision variants in one corpus pass per snapshot and retained depth
100 with descending-score, descending-external-ID tie handling. Every corpus vector-file hash
was rechecked against the complete scan and pinned ingestion receipts before scoring.

Snapshots ran sequentially in fresh WSL workers with one BLAS thread. Observed peak whole-worker
RSS was 200,328 / 281,264 / 304,384 KiB for snapshots 1 / 2 / 3. These include the worker's mapped
pages and Python inputs; they are not index memory or isolated benchmark measurements.

Evidence is under `build/longeval-query-sensitivity-01/`. Each snapshot has two run files, a
completed report, per-query/cohort statistics and resource measurements. The completed job
record SHA-256 is `d1458e4ec4d3e7d27cb6485e940968beb743a83f487e9b6c4bb95b45c4b54420`.

| Snapshot report | SHA-256 |
|---|---|
| snapshot-1/report.json | `1a21cc85fd03ee027f41673180c677cbb290e239fe5aa5cbe5aba587dd91f6ce` |
| snapshot-2/report.json | `7dd52dbad1b19e16aa3e7d26b18a9aa387df0181e5beb8d71cb63e6dbfe06862` |
| snapshot-3/report.json | `8f09f219c5b0cf4765a1b71d8e9d20f97b34873da5e3baca3510140c51cd8be9` |

`summary.json` independently recounts all ranking differences from the six saved run files and
verifies their hashes against those reports. The analysis is reproducible with
`build/summarize-longeval-query-sensitivity-01.py`; its checksum is recorded in the summary.
GPU query provenance is retained in `build/longeval-pace-query-evidence-7107422/`.

## Development decision and remaining work

Use GPU float32 as the candidate configuration for development runs, based on its closer
agreement with the archived document sample, and retain BF16 as a separately declared sensitivity
condition. Do not mix devices, runtimes or precisions within an index-history trajectory.
Neither candidate exactly reproduces the historical vectors/runtime, and no retrospective
acceptance threshold is assigned from the results above.

Before primary experiments: document the reconstruction limits and frozen configuration,
validate development-only relevance/evaluator semantics, implement the bounded full-data ANN
trajectory runner, establish isolated resource budgets and matched operating regimes, and lock
the inferential plan and practical-effect margin before inspecting final test comparisons.
