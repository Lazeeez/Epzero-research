# Does the index matter over time? ANN search in longitudinal dense retrieval

LongEval evaluates retrieval systems over successive snapshots of a growing collection. Its dense
baselines use exact search. Deployed dense retrieval instead runs on an approximate nearest-neighbour
(ANN) index that is built once and then updated in place. This study measures whether conclusions
drawn with exact search transfer to such deployments, and whether keeping an index alive differs
from rebuilding it.

- **Data:** LongEval-Sci 2026. Three cumulative snapshots of 0.87M, 1.32M and 1.66M documents with
  the released Qwen3-Embedding-4B vectors; 100 training queries and 381 held-out test queries.
- **Arms:** exact search, Epzero HNSW, FAISS HNSW and FAISS IVF-Flat. Each index is maintained
  incrementally and also rebuilt from the identical history, with three construction seeds.
- **Protocol:** pre-registered and hash-frozen before any test-query retrieval
  ([FINAL_PROTOCOL_V1.md](FINAL_PROTOCOL_V1.md) and its receipt `FINAL_PROTOCOL_V1_FREEZE.json`).
- **Target venue:** CLEF 2027 conference track.

## Status (22 November 2026)

- **Done and verified:** the primary held-out results, the secondary timing results, and the
  exploratory follow-ups below, including the full-scale insertion-order experiment (24 PACE
  jobs) and the judge pilot.
- **Done since:** the insertion-order experiment with a denser graph on the full collection (6
  jobs; section 15 of the exploratory record). Timing at ef 100, 400, 800 and 1600 has run (3 jobs);
  its evidence is being copied back.
- **Waiting on the advisor:** six open decisions in the
  [extension protocol draft](docs/protocol/EXTENSION_PROTOCOL_DRAFT.md). After them come the
  freeze of the follow-up protocol, grading of the judging pool and the coverage analysis.
- **Manuscript:** a full draft in [paper/](paper/), to be restructured and shortened.

## Results

**Primary results (frozen protocol, 374 test queries, GPT-OSS-120B labels).** The table gives the
change in the ANN-minus-exact nDCG@10 gap from snapshot 1 to snapshot 3, averaged over three seeds.
The test is sign randomization with Holm adjustment; the intervals are bootstrap 95%; the margin is
±0.01.

| Arm (calibrated to 95% neighbour recall) | Change | 95% interval | Holm p | Decision |
| --- | --- | --- | --- | --- |
| Epzero HNSW | −0.0059 | [−0.0129, −0.0006] | 0.117 | inconclusive |
| FAISS HNSW | −0.0007 | [−0.0038, +0.0027] | 0.687 | negligible |
| FAISS IVF, frozen training | −0.0015 | [−0.0037, +0.0005] | 0.291 | negligible |

- **Gap size:** the gap is below 0.012 nDCG@10 at every snapshot.
- **Why Epzero differs:** its calibration never widened its search, while its recall on the test
  queries drifted from 94.5% to 93.5%. Details are in [HELDOUT_RESULTS.md](HELDOUT_RESULTS.md).
- **Index history (RQ1):** an incrementally maintained index is bit-identical to a rebuild from the
  same insertion history in all 27 comparisons (2 HNSW implementations and IVF, 3 seeds, 3
  snapshots).

**Exploratory, after the freeze** ([HELDOUT_EXPLORATORY.md](HELDOUT_EXPLORATORY.md)):

- **New documents.** HNSW misses documents inserted at later snapshots 1.5–1.6 times as often as
  older ones at the same exact rank; IVF shows about 1.15 times. A held-out rebuild in random
  insertion order removes most of this excess. It is an insertion-order effect.
- **Insertion order, full collection.** Building the same 1.66 million documents in reversed
  arrival order brings the pooled miss ratio of the latest arrivals from 1.60–1.75 to 0.97–0.98.
  The paired change is +0.62 [+0.46, +0.80] for FAISS and +0.77 [+0.57, +0.98] for Epzero.
- **Within a query (post hoc).** Compared with first arrivals of the same query at the same
  depth, the latest arrivals are missed 1.28–1.42 times as often when inserted last, 1.00–1.03
  times at a random position and 0.82 times when inserted first. The ratio of about 1.2 that
  survives shuffling in the pooled figures is the difficulty of the queries, not of the documents.
- **Fresh content.** Queries whose answers are mostly new content lose about three times the
  average gap on a fixed-width HNSW index.
- **Reordering does not fix effectiveness.** A shuffled rebuild only spreads the misses across
  document ages.
- **Search width does.** At about 98% recall (ef 800) the gap is 0.002–0.003 at every snapshot. At
  ef 200, HNSW costs about 2.5 ms per query on a single thread; ef 800 has not yet been timed.
- **Calibration size.** In a resampling study, a 50-query calibration set keeps a too-narrow
  search width in 21–22% of draws. Requiring one point of recall above the target cuts that to
  5.5%.
- **Pooling.** Most documents that approximate search substitutes into the top 10 are unjudged. A
  condensed-list check shrinks every gap by roughly half to three-quarters, with the same pattern
  over time.
- **Leniency.** On the 29 test queries that also have human grades, the released LLM labels agree
  with them at a linear-weighted kappa of 0.24; they call 87% of the pairs relevant, the humans 60%.
- **Judge pilot.** A local GPT-OSS-120B judge with the organisers' prompt and parser answered all
  1,080 requests, agrees with the released grades about as well as with itself (87–90% the same
  grade), and grades about 5,000 pairs an hour on one H200
  ([results](docs/protocol/JUDGE_PILOT_RESULTS.md)).

**Development findings (RQ3, RQ4).** Search failures come from limited search coverage, not index
age. Four pre-registered graph-repair methods fail to beat a plain wider search at equal distance
work ([docs/repair/](docs/repair/)).

## Where to start reading

| Document | Contents |
| --- | --- |
| [paper/manuscript-clef2027.md](paper/manuscript-clef2027.md) | Manuscript draft with all results; figures in [paper/figures/](paper/figures/) |
| [FINAL_PROTOCOL_V1.md](FINAL_PROTOCOL_V1.md) | The frozen held-out protocol |
| [HELDOUT_RESULTS.md](HELDOUT_RESULTS.md) | Primary held-out results and their verification |
| [HELDOUT_EXPLORATORY.md](HELDOUT_EXPLORATORY.md) | Post-freeze analyses: timing, shuffled order, the ef sweep, external validation, the judging pool, the insertion-order pilot and the full-scale insertion-order experiment |
| [../../reports/lajat-longeval-research-statement.pdf](../../reports/lajat-longeval-research-statement.pdf) | Statement of research (for submission): what the study asks, how it is designed, what it has found and what remains; the manuscript holds the method in detail |
| [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) | Chronological log of every decision and PACE job |

## Layout

| Path | Contents |
| --- | --- |
| `*.py` | The experiment code (module map below) |
| `tests/` | Unit and regression tests (pytest) |
| `paper/` | Manuscript, figure script and rendered figures |
| `docs/protocol/` | The approved protocol candidate and the earlier draft; for the follow-up, the judge-validation thresholds, the judge pilot's results, the extension protocol draft and the design record of the full-scale insertion-order experiment |
| `docs/data/` | Release audit, vector ingestion, encoder and query-encoding validation |
| `docs/development/` | Development experiments on training queries: trajectories, failures, IVF, cost |
| `docs/repair/` | Pre-registered repair screens (RQ4), including the NGFix/RFix reproduction |
| `docs/history/` | Superseded plans, the synthetic-pilot documentation and an earlier draft |
| `trajectory-development.json`, `ivf-development.json` | Index configurations (frozen) |
| `insertion-order-full*.json` (including `-m32`, the denser graph), `insertion-order-draw*.json` | Aggregates of the full-scale insertion-order experiment and of its three half-size draws (no identifiers): widths, contrasts and decile curves, and the counts of each full-collection build |
| `insertion-strata*.json` | Within-query miss ratios (no identifiers): the five collections, the other search widths, GloVe and SIFT, the held-out arms, and the denser graph (`-m32-ef*`) |
| `insertion-pilot-*.json` | Aggregates of the insertion-order pilot (no identifiers): the first run (`summary`), other search widths and graph settings (`robustness`), misses by insertion decile (`position`), and the replication on GloVe and SIFT (`public-*`) |
| `requirements.txt`, `encoder-requirements.txt` | CPU research and GPU encoder dependencies |
| `../../scripts/pace/LONGEVAL*.md`, `longeval-*` | PACE runbooks, job scripts and packagers |

## Module map

| Area | Modules |
| --- | --- |
| Data and release audit | `release.py`, `dense.py`, `storage.py`, `mapped.py`, `documents.py` |
| Query encoding | `encoder.py`, `query_diagnostic.py`, `query_sensitivity.py`, `representation.py`, `test_queries.py` |
| Exact search and evaluation | `oracle.py`, `evaluation.py`, `paired_inference.py`, `relevance_analysis.py`, `development_relevance.py` |
| Index trajectories | `backends.py`, `trajectory.py`, `ivf_trajectory.py`, `ivf_verification.py`, `runner.py` |
| Held-out protocol | `heldout.py`, `heldout_analysis.py`, `freeze_protocol.py` |
| Follow-up protocol | `freeze_extension.py` ([draft](docs/protocol/EXTENSION_PROTOCOL_DRAFT.md)) |
| Post-freeze work | `heldout_cost.py`, `heldout_extension.py`, `heldout_exploratory.py`, `heldout_robustness.py`, `heldout_pool.py`, `insertion_pilot.py`, `insertion_strata.py` |
| Follow-up: human audit | `audit.py` (blinded annotation pages for dropped and replacement documents, and the analysis of returned grades) |
| Follow-up: local relevance judge | `judge.py`, `judge-requirements.txt`, `judge-kernels.txt`, `judge-model.json`; `coverage.py` for the validation sample, the three validation measures and the coverage estimate ([runbook](../../scripts/pace/LONGEVAL_JUDGE_PILOT.md), [thresholds](docs/protocol/JUDGE_VALIDATION_THRESHOLDS.md)) |
| Development diagnostics | `failure_diagnostic.py`, `trace_phases.py`, `work_budget.py`, `native_cost.py`, `native_cost_verification.py`, `retention_search.py`, `retention_diagnostic.py`, `feasibility.py` |
| Repair screens (RQ4) | `frontier_search.py`, `frontier_pilot.py`, `witness_repair.py`, `witness_pilot.py`, `region_margin.py`, `ngfix_*.py`, `ngfix_driver.cpp`, `standard_graphs.py`, `standard_graph_pilot.py`, `pseudo_queries.py`, `pseudo_query_gates.py` |
| Synthetic pilot | `contracts.py`, `fixture.py`, `pilot.json`, `__main__.py` ([docs/history/PILOT.md](docs/history/PILOT.md)) |

## Running the code

The tests need Python 3.11 or later. The requirements install the released Epzero engine (0.5.0) from PyPI. From the repository root:

```sh
python -m pip install -r research/longeval/requirements.txt
python -m pytest -q research/longeval/tests
```

Some tests skip under certain conditions:

- Tests marked `linux_native` run only on Linux, because those modules require the validated Linux
  native build.
- The `hnswlib` comparison tests run only if `hnswlib` is installed.

The full experiments run on the PACE cluster: CPU jobs on Gold 6226 nodes, and GPU jobs for query
encoding and the local relevance judge. Each job is a self-contained package that verifies its own checksums and inputs before
downloading any data. The runbooks in `scripts/pace/` give the exact commands. The held-out ones are
`LONGEVAL_TEST_QUERIES.md`, `LONGEVAL_HELDOUT.md`, `LONGEVAL_HELDOUT_COST.md` and
`LONGEVAL_HELDOUT_EXT.md`.

To check that the frozen code is unchanged, run `python scripts/verify_freeze.py`. To run the tests, the
freeze check and the synthetic pilot in one go, run `bash scripts/verify.sh`. [REPRODUCE.md](../../REPRODUCE.md)
describes every level of reproduction, up to the full study.

## Data policy

LongEval data are not redistributed here: document vectors, query texts and vectors, judgments, and
run files. All of them are downloaded from the pinned official release and kept under `build/`,
which git ignores. The repository holds code, configurations, protocols, aggregate results and
figures only.
