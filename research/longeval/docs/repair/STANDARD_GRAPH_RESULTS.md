# Native competitor harness check — 2026-11-18

The new additive adapters now support **Epzero, FAISS HNSW and hnswlib0.8.0** under
one explicit IP-query protocol. Nine graphs (three implementations x seeds11/29/47)
and 3,600 query conditions completed and passed independent ranking/score checks.
These are small-sample local baseline measurements, not a full-scale method result.

## Protocol

- Same 8,192-row snapshot-1 sample, 2,560 dimensions, float32, uncompressed IP,
  insertion order, k100, M16, efConstruction100, and 100 training queries.
- Local WSL on Intel i5-13420H, process pinned to logical CPU0, one search/build
  thread. This is not an exclusive PACE allocation or a controlled compiler comparison.
- Effort grid 100/200/400/800. For each implementation/seed, choose the smallest ef
  attaining mean calibration recall .95 or .99. Persist selection before evaluating
  development queries. All methods/seeds select ef200 and ef400 respectively.
- Each query gets one immediately preceding warmup and three native binding calls.
  Output conversion, score verification and state hashing are outside timing. Backend
  order rotates by query/grid/seed; construction order rotates by seed.
- Float64 references and original external row-label mapping are checked. Every
  native repeat must return identical scores and labels. Serialized graph hashes
  are identical before/after after restoring search settings. The seed11 Epzero
  state reproduces the earlier pilot index byte-for-byte.

## Descriptive development measurements

Each range spans the three seeds. Time is the cohort mean of each query's median
of three calls; it is not a service throughput measurement.

| Calibration target / selected ef | Implementation | Achieved development recall | Native call time, ms |
| --- | --- | ---: | ---: |
| .95 / 200 | Epzero | 97.54–97.62% | 1.011–1.040 |
| .95 / 200 | FAISS HNSW | 97.60–97.88% | 0.941–0.959 |
| .95 / 200 | hnswlib0.8.0 | 97.52–97.64% | 1.133–1.180 |
| .99 / 400 | Epzero | 99.42–99.44% | 1.642–1.687 |
| .99 / 400 | FAISS HNSW | 99.34–99.54% | 1.532–1.552 |
| .99 / 400 | hnswlib0.8.0 | 99.42% | 1.820–1.848 |

FAISS has the lower measured times at these selected settings. These comparisons
do not establish universally matched achieved recall: the coarse grid overshoots
the targets and the achieved recalls differ. All .95-target development conditions
exceed .95, but individual-query minima range from .87 to .91 at ef200. A mean
target is not a per-query guarantee.

Serialized indexes are approximately 85.1 MB each; this does not measure process
RSS. Build wall times are approximately 4.15–4.28s for FAISS/Epzero and 4.85–5.02s
for hnswlib in this run. Fixed M/efConstruction is only a controlled harness check;
the paper needs memory/build-budget Pareto tuning and much larger data.

## Verification and limits

Six native adapter tests pass for score semantics, nonsequential labels, repeatability,
invalid data and padding rejection. An independent verifier recounts all 3,600
conditions, recomputes recall, timing summaries and calibration choices, and checks
every returned score against float64 products. Maximum score discrepancy across
this check and the repair screen is 1.81e-7. Native timing measurements themselves
are not independently replicated by an arithmetic audit.

The hnswlib package is isolated under `build/longeval-competitors-py`; existing
experiment packages are unchanged. The cached Linux wheel and loaded binary hashes
are recorded. The comparison pins 0.8.0 and does not claim coverage of the latest
upstream changes or matched compiler flags. A PACE package must build hnswlib for
the actual compute-node CPU, as the existing repository benchmark instructions require.

No repaired-graph native implementation is timed. These results do not support a
new-method speedup. The previously prepared full-data native timing package remains
pending returned results and includes only Epzero/FAISS; it does not include these
new adapters, hnswlib or either experimental method.

Artifacts: `build/longeval-standard-graphs-01/`,
`build/longeval-standard-graph-tests-01.xml`, and
`build/longeval-method-screen-02-verification.json`.
