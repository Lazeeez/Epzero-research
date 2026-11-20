# Research completion plan

The primary aim, updated at the author's request on 2026-11-18, is a **new method paper
with competitive performance**. The existing empirical study supplies the diagnosis
and controls. Follow [METHOD_RESEARCH_PLAN.md](../repair/METHOD_RESEARCH_PLAN.md) for method
development, ablations and competitor promotion gates. Publication is an external
review outcome; completed experiments do not establish novelty or acceptance.

Before the final evaluation sequence below, develop a deployable policy, distinguish
it from prior methods, test it at matched work, and establish native performance
against tuned competitors. Keep the held-out retrieval outcomes untouched throughout
this selection. The first candidate, overlap-guided frontier resumption, failed its
small-pilot matched-work promotion gate; preserve that result and investigate the
next justified mechanism. A second calibration-only witness-repair screen also failed
to show meaningful development transfer. Avoid further ad-hoc tuning on these 100
queries. Establish a structural signal or a larger disjoint learning population first.
A novel method with a real-data advantage is still needed. Native competitor adapters
now include hnswlib; a three-seed small-sample comparison is validated.

A new 1,800-trace audit identifies residual neighborhood coverage as the main
target at the selected work budget. Published NGFix/RFix directly overlaps that
mechanism, so it is now a required comparison for repair claims. Its four-control
native screen is now independently verified from job7141656:1,600 real conditions
and185,600 real/synthetic score checks. NGFix's small development gain is much less
than its fitted-query gain; fitting and search-setting selection must be separated
before interpreting performance at a target recall. The381 final held-out outcomes
remain untouched. This is competitor infrastructure, not a third successful method
candidate. See [TRACE_PHASE_RESULTS.md](../development/TRACE_PHASE_RESULTS.md) and
[NGFIX_RESULTS.md](../repair/NGFIX_RESULTS.md). Full-data and independently trained competitor
comparisons remain required after the revised protocol and distinct method are ready.

The revised [disjoint-role protocol](../repair/NGFIX_DISJOINT_PLAN.md) completed as job7148412;
all24 selected points meet development mean recall. See [results](../repair/NGFIX_DISJOINT_RESULTS.md)
for the remaining raw-state identity/initialization limits before larger comparisons.
A separate fit-only
[regional-bound feasibility check](../repair/REGION_MARGIN_RESULTS.md) rejected that proposed
rule without a retrieval run. This does not close the novel-method requirement.

## Supporting evidence and final evaluation sequence

Development relevance, IVF controls and the original HNSW native-cost comparison
are now complete. Jobs7139467/8/9 reproduce all18 baseline states and supply54 native
timing cohorts; see [NATIVE_COST_RESULTS.md](../development/NATIVE_COST_RESULTS.md). The sequence
below records the larger study requirements. Final held-out evaluation remains
blocked on establishing and selecting a novel method with competitive performance,
not on obtaining these original baseline timings.

1. Score the existing development rankings against the two training judgment sources,
   with independent evaluator parity. Training judgments belong to snapshot 1. Scores
   for later snapshots against those same labels are historical-label diagnostics,
   **not contemporaneous temporal effectiveness**. Do not inspect test retrieval outcomes.
2. Execute three IVF controls at seeds 11, 29, 47: frozen-training incremental,
   frozen-training rebuild with identical chronological insertion, and current-snapshot
   retraining plus rebuild. Bound training data, declare all probe settings, retain
   target misses, and measure native search work and latency separately from harness costs.
3. Compare the resulting mechanisms and costs with HNSW. Native latency comparisons
   require a common measurement protocol/hardware; old instrumented wall times cannot
   substitute. Complete a primary-source related-work comparison and choose a narrow
   claim supported by the development results. A null result is an allowed conclusion.
4. Freeze a versioned final protocol before any test retrieval evaluation: representation,
   arms/settings, judgment families and populations, primary contrasts, practical-effect
   margins, query-level paired inference and multiplicity rules. Test query IDs/judgment
   coverage were previously audited; test retrieval outcomes have not been used for tuning.
5. Encode the held-out queries, generate exact references, run the frozen experiments,
   and independently verify outputs. Use snapshot-specific judgments. Report changing
   populations and a fixed panel separately; do not pool automated and human judgments.
6. Add a declared replication setting if making a general claim; otherwise explicitly
   scope the paper to this collection/representation and document the generality limit.
7. Produce the manuscript, figures, reproducibility instructions, artifact checksums and
   limitations. Select a venue based on the actual contribution and its current rules.
   Submission requires author details and the author's approval of the finished manuscript.

## Constraints

- Preserve the original 127 source files and all sealed experiment supplements.
- Keep existing results and verification receipts immutable. Add new analysis outputs.
- All large PACE downloads/work files remain on compute-node temporary disk, with no
  shared-storage fallback. Persist bounded outputs only.
- Use calibration queries for search selection; development queries for exploratory
  analysis. No post-hoc target, seed, or query filtering to make findings favorable.
- Do not claim a causal decomposition from subtracting aggregate metric values.
- PACE execution currently uses user-submitted commands; prepared jobs must include a
  tested package, checksums, exact submission instructions and retrieval instructions.
