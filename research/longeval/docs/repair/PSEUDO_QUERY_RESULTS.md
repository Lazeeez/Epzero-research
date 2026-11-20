# Pseudo-query bridging — stage-2 sample gates, 2026-11-19

**Result: rejected at gate 4.** Encoder pseudo-queries look like real queries and their
bridge edges reduce answer-set fragmentation, but the extra edges cost more distance
work than they recover. At equal work, every encoder arm is slightly *worse* than the
plain graph, and a corpus-only control (documents as their own queries) does better.
Under the rule fixed in [PSEUDO_QUERY_PLAN.md](PSEUDO_QUERY_PLAN.md), this direction ends.

## Inputs

- PACE job 7151090 (A40, FP32, TF32 off; 36 min). Encoder gate: minimum cosine
  0.99999999999 against the A100 job-7107422 query vectors. The job verified both sets;
  local checksums match for every returned file. Set B (131,040 vectors) stays on PACE.
- **Deviation (defect):** package 01 indexed the sorted final ID map with snapshot-1 row
  positions. Set A titles therefore come from 8,192 corpus documents, of which only 40 are
  in the sample. Here pseudo-queries act as a sample of the query distribution, not as each
  sample document's own query. `prepare` is corrected (`sample_ids`); Set B was unaffected.
- Graph: the saved Epzero sample index (8,192 snapshot-1 vectors, M=16, efC=100, seed 11).
  Python replay matches native Epzero results and distance counts in all 500 base checks.
- 50 development queries, exact in-sample top-100; nothing selected on them.

## Gates

| Gate | Result | Measure |
| --- | --- | --- |
| 1 In-distribution | pass | Dev-query top-1 IP: documents 0.495, titles 0.520, keywords4 0.556 |
| 2 Coverage | pass | Pseudo-query top-100 unions cover 100% of dev answer occurrences (98.8–98.9% at top-10); weak at sample scale |
| 3 Fragmentation | pass | Mean components of dev top-100 sets: base 6.80, title 5.88, keywords4 5.84, both 5.74; docself 5.96, random 6.62 |
| 4 Equal work | **fail** | No encoder arm has a positive equal-work gain; docself beats all of them |

Bridging rule (fixed before search): for each pseudo-query, take its exact top-32; join
each answer component not containing the top-1 answer through its best member to the top-1
answer, both directions, at most 4 added edges per node. Controls: `docself` uses each
sample document as its own query; `random` adds the same number of edges as keywords4 at
uniformly random endpoints under the same budget.

## Gate 4 detail (development mean over 50 queries, k=100)

Equal-work gain = arm recall minus base recall interpolated at the arm's mean distance count.

| Arm | Added edges | ef | Distances | Recall | Min | Equal-work gain (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0 | 100 / 150 / 200 | 1,321 / 1,770 / 2,159 | 93.40 / 96.24 / 97.56 | 78 / 86 / 89 | — |
| title | 8,350 | 100 / 150 / 200 | 1,468 / 1,959 / 2,379 | 94.44 / 96.66 / 97.84 | 82 / 86 / 89 | −0.09 / −0.39 / −0.27 |
| keywords4 | 9,164 | 100 / 150 / 200 | 1,487 / 1,988 / 2,414 | 94.42 / 97.00 / 98.16 | 79 / 83 / 91 | −0.25 / −0.13 / −0.04 |
| both | 10,956 | 100 / 150 / 200 | 1,505 / 2,007 / 2,436 | 94.72 / 96.90 / 98.08 | 82 / 86 / 89 | −0.10 / −0.28 / −0.17 |
| docself | 5,402 | 100 / 150 / 200 | 1,411 / 1,883 / 2,295 | 94.40 / 96.76 / 98.00 | 81 / 84 / 92 | +0.31 / +0.04 / +0.10 |
| random | 9,164 | 100 / 150 / 200 | 1,506 / 2,026 / 2,478 | 93.70 / 96.38 / 97.76 | 82 / 86 / 89 | −1.12 / −0.85 / −0.60 |

The base graph has 149,319 directed layer-0 edges; keywords4 adds 18,328 directed edges
(12%). At equal ef, bridging raises recall by about 1 pp at ef 100, but each expansion
now scores more neighbors, so distance work rises 11–13%. Targeted edges are far better
than random edges, so the fragmentation signal is real, but it is not cheaper than simply
searching wider on this sample. The query-distribution advantage (gate 1) did not
translate into better edges than the documents' own neighborhoods (docself).

## Interpretation and limits

The sample is easy (base 97.6% at ef 200); the full corpus has more severe fragmentation and
could behave differently. That is not grounds to continue: the gate was fixed in advance and
the encoder-specific contribution lost to a cheaper corpus-only control here. Docself's small
positive gain (+0.04–0.31 pp) resembles existing graph-refinement methods and is not a novel
method. Search cost is Python-replay distance counts, not native latency.

Evidence: `build/longeval-pseudoquery-gates-01/report.json` and `per-query.jsonl`
(2,000 development search rows); code `pseudo_query_gates.py`, 5 tests. Inputs:
`build/longeval-pseudoq-7151090-setA.tar.gz`, SHA-256
`5cb4c257131f2c8012731c78d67457e8a60204612168c6c73784f971bd23c106`.
The 381 held-out query retrieval outcomes remain untouched.
