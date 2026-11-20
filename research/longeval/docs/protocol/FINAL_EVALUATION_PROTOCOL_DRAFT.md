# Final evaluation protocol — DRAFT, not frozen

**2026-11-18 scope change:** the primary objective is now a new method paper with
competitor performance. This older empirical protocol is supporting material and is
not sufficient to authorize a final method evaluation. Before freeze, add the selected
deployable algorithm, direct method baselines, calibration resources, matched-quality
cost curves, ablations, native implementation identity, external replication settings
and tail-quality margins from [METHOD_RESEARCH_PLAN.md](../repair/METHOD_RESEARCH_PLAN.md).
No method has yet passed those gates. No held-out retrieval outcomes were used.

This document does not authorize interpreting development findings as held-out evidence.
Finalize, timestamp and hash the protocol plus executable configurations before producing
test retrieval outcomes. Freeze only after IVF feasibility and common native-cost
measurement are validated. Amendments after that point require a separate version and
must identify which test outcomes were already visible.

## Representation and scope

Candidate primary representation: original archived document vectors, unchanged;
Qwen3-Embedding-4B query reconstruction with the pinned baseline prompt/model revision,
CUDA float32 and TF32 disabled, as in the verified training-query diagnostic. This is
a declared reconstruction, not historical runtime reproduction. Exact and approximate
arms must use the same query vectors. The float32/BF16 development sensitivity findings
are disclosed; no representation choice will depend on test relevance scores.

Use all three audited cumulative snapshots. The intervention is collection growth with
fixed shared text/vectors; there are no deletions, rewritten documents or encoder updates.
General claims about arbitrary mutable collections are outside this design.

## Populations and labels

- The 100 training queries remain split into 50 calibration and 50 development IDs.
  Search selection uses calibration neighbor recall only. All 381 separate test queries
  are available for final retrieval; query IDs and judgment coverage were audited earlier.
- For each snapshot and judgment family, report the explicitly defined positive-judgment
  population and its denominator. Do not treat absence of labels as observed zero utility.
- Longitudinal paired estimates use a fixed query-ID intersection for that judgment family.
  Report the available populations separately to expose selection effects. Existing coverage
  audit reports common-positive panels of 10 DCTR, 80 raw clicks, and 374 GPT-OSS-120B
  queries; verify those identities from the pinned files when freezing.
- Keep click-derived, LLM-generated and human labels separate. Human labels are available
  for 29 queries at snapshot 3, not a three-snapshot panel. They support a limited final-
  snapshot sensitivity analysis, not temporal human validation.
- **Unresolved before freeze:** designate the primary label family and population based on
  the scientific claim and measurement validity, not observed test effectiveness. A larger
  automated panel is not a substitute for reliable human judgment. If the small click panel
  cannot support the intended precision, narrow the claim or obtain an external replication.

## Outcomes and contrasts

Primary effectiveness candidate: nDCG@10, linear gains and explicit deterministic ties.
Secondary: relevant-document Recall@100, judged coverage@10, ANN neighbor recall@100,
per-query failure tails, query latency and actual search work. Report signed ANN-minus-
exact differences; exact similarity rankings are not a relevance upper bound.

Candidate predeclared contrasts, finalized after development IVF results:

1. Incremental versus identical-history/same-training rebuild: exact state/ranking equality
   where expected, with positive maintenance cost. Equality checks are not significance tests.
2. Frozen search setting versus calibration-adjusted search on the same index, including
   additional native query cost. Offline prefixes alone cannot establish latency benefits.
3. Frozen versus refreshed IVF training at a common probe setting, then at a common declared
   neighbor-recall target, reporting work differences rather than calling nprobe matched cost.
4. Change in the signed ANN-versus-exact relevance gap from snapshot 1 to snapshot 3, on
   the same judged query panel. This difference-in-differences is a paired metric contrast,
   not a causal decomposition into model and index drift.

Freeze the exact arm list, seed list, settings, primary comparisons and directionality.
Keep all seeds 11/29/47; do not select the best construction seed. A selected-case deferred-
candidate intervention is diagnostic and is not a deployable primary algorithm.

## Inference and practical importance

Use the query as the resampling unit. Preserve each query's paired arms, snapshots and
construction seeds together. Average the prespecified seeds within each query before
estimating the query mean contrast; three seeds do not create three independent queries.
Report the individual seed results as sensitivity evidence.

Candidate inference: 10,000 paired query bootstrap samples with a fixed recorded RNG seed,
95% percentile intervals, and two-sided paired sign-randomization tests with 100,000 draws.
Apply Holm adjustment across the explicitly frozen primary comparison family. Distinguish
nominal confidence intervals from multiplicity-adjusted testing. Small fixed panels require
prominent uncertainty reporting; a non-significant result does not establish equivalence.

Before freeze specify an absolute nDCG practical-effect margin and a cost criterion. A
possible planning value is 0.01 absolute nDCG, but this is a research design choice that
needs justification, not a fact inferred from development results. No equivalence or
maintenance recommendation is allowed without both a declared margin and measured costs.

## Resource protocol and artifact gate

Run native comparison arms on the same declared CPU type, thread count, query batching,
warmup/repetition policy and affinity. Record native-call samples separately from build,
training, audit and instrumentation time. Whole-worker RSS must not be labelled native
index allocation. Separate distance computations from coarse search and graph expansions.

Verify source/input hashes, sample IDs, exact references, result membership, exported
scores, evaluator parity and control equality. Freeze metric/population/inference code
alongside the configuration. Invalid execution requires a documented repair/re-execution;
unfavorable valid outcomes remain in the paper.

## Required items before changing this draft to frozen

1. IVF results are now returned and independently verified. Retain both probe regimes,
   including seed 29's improvement at the original 256-probe setting.
2. Common native-cost protocol and its tested implementation.
3. Primary judgment population, primary contrast family and justified practical margin.
4. Executable held-out encoding/oracle/trajectory and inference pipeline.
5. Primary-source novelty comparison, scoped claim, and explicit replication decision.

No test-query retrieval evaluation has been performed by this continuation.
