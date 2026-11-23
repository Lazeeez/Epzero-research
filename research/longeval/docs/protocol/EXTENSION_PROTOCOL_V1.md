# Extension protocol, version 1

**Drafted 21 November 2026; decisions recorded and finalised 23 November 2026; frozen by
`freeze_extension.py` (receipt `EXTENSION_PROTOCOL_V1_FREEZE.json`).** It follows revision 3
of the author's follow-up plan. The advisor was unavailable, so the six open points were decided by
the author on 23 November 2026; each decision, its reason and what was already known when it was
made are recorded in section 5. The protocol cannot be frozen while any point remains open
(`freeze_extension.py` refuses).

`FINAL_PROTOCOL_V1` is frozen and stays as it is. This extension covers experiments that were
designed after its results were seen. They reuse the same 381 test queries, so every result under
this extension is reported as a **prespecified follow-up**: defined here before its own outcome
exists, but not independent of the held-out results.

## 1. Questions

The held-out study left three explanations open for the gap between approximate and exact search
and for its change over time:

1. **Judgment coverage.** Most documents that an approximate index substitutes into a top 10 are
   ungraded and count as non-relevant.
2. **Insertion order.** An HNSW index built by appending under-retrieves its latest arrivals.
3. **The representation.** Everything rests on one embedding model.

## 2. What has already been seen

Freezing this protocol does not make these outcomes unseen. They are prior evidence.

| Outcome | Where |
| --- | --- |
| Primary and secondary held-out results | `HELDOUT_RESULTS.md` |
| Exploratory analyses of the held-out evidence: arrival groups, replacements, condensed lists, fresh-content queries, shuffled order, ef sweep, baseline reproduction, leaderboard counterfactual, per-query tail, calibration size and margin, judging pool and carried grades, LLM against human grades, gap change by search width | `HELDOUT_EXPLORATORY.md`, sections 1–9, 11, 12 |
| Insertion-order pilot on a 300,000-document sample: three orders, four search widths, two graph settings, insertion deciles, and two public benchmarks | `HELDOUT_EXPLORATORY.md`, section 10 |
| Judge pilot on 270 prompt-development pairs | `docs/protocol/JUDGE_PILOT_RESULTS.md`: job 7219729, evidence SHA-256 `4013158486b7a7913d242de757f73d3b3c3acc3c5fd2114b3d6f84dfa96cfcfa` |

## 3. Experiments

### E1. Judgment coverage

**Assessor.** GPT-OSS-120B run locally with the organisers' prompt and answer parser (`judge.py`,
pinned by tests). The serving stack and model revision are those of the judge pilot
(`judge-requirements.txt`, `judge-kernels.txt`, `judge-model.json`).

- **Decided (1): temperature 0.** The pilot (`JUDGE_PILOT_RESULTS.md`) gives a repeat agreement of
  0.874 with no sampling parameter and 0.915 at temperature 0, a paired difference of +0.041
  [−0.004, +0.085], and no difference in agreement with the released grades.
- **Answer cap:** 4,096 tokens. The longest of the pilot's 1,080 answers used 773.
- **What the released grades are.** The organisers sent no sampling parameter, so each released
  grade is one sample at their provider's default. The pilot's agreement between two repeats of
  the same setting is therefore the ceiling for any agreement with the released grades, and the
  validation result is read against it.
- **One configuration per grading folder.** Before the first request, the grading run records
  the checksums of the inputs in their order, the model revision, the prompt, the parser, the
  request parameters, the answer cap and the server settings. A run that resumes under anything
  else is refused (`coverage.grade`).
- **Usable grades.** A pair has a grade only if an answer parsed. An answer that does not parse
  is asked again with the same request, at most twice more; every attempt is kept and the first
  parsed answer is the grade. A pair that never parses has no grade. The organisers' pipeline
  wrote grade 0 in that case; here the pair is reported as missing.
- **Validation.** As recorded in `JUDGE_VALIDATION_THRESHOLDS.md`, on 1,000 officially graded
  pairs that exclude the 270 pilot pairs. **Decided (2):** the thresholds of its section 3 are the
  gate (binary agreement ≥ 0.90, linear-weighted kappa ≥ 0.80, nDCG@10 difference ≤ 0.02 at every
  snapshot). The stricter bar of its section 5 (binary agreement ≥ 0.932 and kappa ≥ 0.835, the
  agreement of the released 20B labels) is reported as a second tier, not a gate: a judge that also
  meets it is described as at least as close to the official grades as the released 20B labels; one
  that meets only the gate is described as compatible. Agreement with the released human grades is
  reported beside both, with no threshold.

**Pool.** The top 10 of exact search and of the contributing runs, at all seeds and snapshots,
graded once per distinct query–document pair, including pairs that already have an official
grade.

- **Decided (3): primary arms plus the shuffled-order runs and the ef sweep**, 6,298 pairs (1,138
  never graded), **plus the top-10 substitutions of the full-scale order rerun** (E2b) on the test
  queries at ef 200, all three orders and seeds, added before grading starts. The arrival-order
  analyses (`docs/arrival-bias-research.md`) need relevance for the documents each order drops and
  puts in their place; the extra pairs cost about an hour or two of grading.

**Estimates.** For each query, arm and snapshot, G is the nDCG@10 of the approximate run minus
that of exact search.

| Estimate | Grades | Measures |
| --- | --- | --- |
| Official | Released grades, as in the frozen analysis | The benchmark as published |
| Condensed | Released grades, ungraded documents removed from every ranking | Sensitivity to ranking graded documents only |
| Official plus filled | Released grades, with local grades where none exists | The effect of supplementing the release |
| Local, masked | Local grades, hidden wherever the release has no grade at that snapshot | The same assessor under the released coverage |
| Local, full | Local grades for the whole pool | Effectiveness with full coverage of the pool |

- **Primary coverage estimate.** C = G(local, full) − G(local, masked), per query and seed, with
  seeds averaged within each query. It is reported at each snapshot, and as C at snapshot 3 minus
  C at snapshot 1, with paired query-bootstrap intervals. It uses one assessor throughout, so it
  does not depend on the validation outcome.
- **Normalization.** The ideal DCG of a query at a snapshot is computed over the pool documents
  of that query at that snapshot, under the grades visible in that estimate. A query whose ideal
  DCG is zero scores 0 and is counted; the count is reported.
- **Scoring universe.** The documents that count for a query at a snapshot are written to a file
  when the pairs are drawn, from the contributing runs chosen above, and the analysis scores on
  exactly that file. The contributing runs and the checksum of the universe are both reported.
- **Completeness.** The primary estimates need a usable grade for every validation pair and
  every pair of the scoring universe. If one is missing, the analysis stops; a partial run can
  only produce a report marked as a diagnostic, with the missing pairs counted by snapshot and
  by whether the release graded them.
- **Panel.** The 374 queries of the frozen primary panel.

### E2. Insertion order at full scale

Design, datasets and definitions are fixed in `scripts/pace/LONGEVAL_INSERTION_ORDER.md`: both
implementations, three seeds, three insertion orders, on all of snapshot 3 and on three
stratified draws of snapshot-1 size.

- **Primary order effect.** The as-arrived miss ratio minus the reversed miss ratio at ef 200, for
  snapshot-3 arrivals against snapshot-1 arrivals, on the full snapshot: per implementation, as
  the mean over three seeds, with a paired query bootstrap.
- **Decided (4): snapshot 3 against snapshot 1.** This was the recommendation written in this
  draft on 21 November 2026, before the full-scale jobs returned. The decision itself is made on 23
  November 2026, *after* the full-scale outcomes were seen (as arrived minus reversed, rank-standardized:
  +0.77 [0.58, 0.98] Epzero, +0.62 [0.46, 0.80] FAISS), so it is reported as adopting the
  pre-outcome recommendation, not as a blind choice. The pooled contrast (snapshots 2 and 3) is
  reported beside it.
- **Secondary.** The same contrast at ef 100, 400 and 800; the decile curve of the shuffled
  build; incoming links by insertion decile; the three draws, with draw variation reported
  separately from seed variation.

### E2b. Order rerun with returned IDs, and confirmation queries

Design, checks, analyses and predictions in `scripts/pace/LONGEVAL_INSERTION_ORDER_IDS.md`
(recorded 23 November 2026, before submission): the full-scale M 16 builds again, saving returned
IDs, and searching the **100 training queries as a confirmation set** never used for any
arrival-order outcome. **Primary for the follow-up:** on the confirmation queries at ef 200, the
within-query rate difference (snapshot 3 against 1), as arrived minus reversed, per implementation;
predicted positive with a 95% interval excluding 0. Paired target-level analysis
(`order_pairs.py`) and the relevance consequence are reported beside it.

### E3. Human audit

**Replaced for the paper (23 November 2026)** by the arrival-order audit of
`scripts/pace/LONGEVAL_INSERTION_ORDER_IDS.md`, analysis D: 50 documents of snapshot 3 and 50 of
snapshot 1 dropped from an exact top 10 by the as-arrived or reversed build, and the 100 documents
put in their place, blind to role, arrival group and order. The paper's question is now whether
the documents an order costs are relevant, which the sample below cannot answer. The sample below
(drawn 21 November) was never graded and stays on disk; it is not used. The rest of this section
(estimand, outcome, annotators) applies to the replacement audit.

Original design (superseded): 200 pairs, blind to role: 100 documents dropped from an exact top 10 and 100 documents put in
their place, drawn over the three primary arms, their seeds and the three snapshots (`audit.py`).
Pairs that play both roles are excluded. Grades use the 0–2 scale of the organisers' prompt, and
an annotator can abstain on a pair; abstentions are counted and left out of the grade statistics.
The sample was drawn on 21 November 2026, before any local judge output existed.

- **Estimand.** The mean grade over distinct pairs of each role. It is not the rank-weighted loss
  over all substitutions: the kit records how often a pair occurs, not at which ranks.

- **Outcome.** The mean human grade by role and their difference, with a query-clustered
  bootstrap interval.
- **Also reported.** Agreement with the official grades where one exists, and agreement between
  two annotators on a 50-pair overlap. **Decided (5): the author grades all 200 pairs blind to role,
  arrival group and order, and a second annotator is actively sought.** If a second person grades
  the 50-pair overlap, blind, before the author's grades are compared with any role label, the
  addition is recorded with its date. Otherwise the single-annotator limitation is stated.

### E4. Second encoder

**Decided (6): in scope**, with BAAI/bge-base-en-v1.5 (768 dimensions, a different model family and
size from Qwen3-Embedding-4B), chosen on 23 November 2026 before any of its outcomes exists.

- **Revision and files.** Pinned to one Hugging Face commit, with file sizes and hashes, in
  `docs/protocol/E4_SECOND_ENCODER.md`, written before the first document is encoded.
- **Text.** The same document text as the Qwen compatibility reconstruction (`encoder.read_records`
  input). BGE accepts at most 512 tokens, so longer documents are truncated; the share truncated and
  their token counts are recorded and reported. Queries use BGE's published retrieval instruction;
  documents use none.
- **Design.** The held-out design of the frozen study (same queries, panel, arms, seeds,
  calibration, labels), with the BGE vectors in place of the released ones, and the arrival-order
  reversal of E2 on BGE vectors.
- **Margins.** The frozen ±0.01 for the within-encoder tests (one Holm family, section 4) and ±0.01
  for the paired contrast between encoders of D.

## 4. Inference

- Seeds are averaged within each query before resampling.
- Intervals are 95% paired query-bootstrap intervals with 10,000 draws, as in the frozen
  analysis, unless a section above states otherwise. They are nominal per contrast.
- No contrast of this extension is a confirmatory test with an error rate controlled across the
  extension. If the second encoder is in scope, its three within-encoder arm tests form one Holm
  family at 0.05, with the frozen ±0.01 margin.
- A smaller, larger, reversed or imprecise effect is a valid outcome and is reported as such.

## 5. Open decisions

None remain. Decided by the author on 23 November 2026, the advisor being unavailable:

| # | Point | Decision | Outcome already known when decided? |
| --- | --- | --- | --- |
| 1 | Judge sampling setting | Temperature 0 | Judge pilot only (no study outcome) |
| 2 | Judge thresholds | Section 3 thresholds as the gate; the 20B-agreement bar as a reported second tier | No |
| 3 | Judging pool | Primary arms, shuffled-order runs and ef sweep (6,298 pairs), plus the E2b order rerun's top-10 substitutions | No |
| 4 | Reversal test of E2 | Snapshot 3 against snapshot 1 (the 21 November recommendation); confirmation on the 100 training queries (E2b) | **Yes** for the test queries: the full-scale E2 outcomes were seen. **No** for the confirmation queries |
| 5 | Human audit and second annotator | The arrival-order audit replaces the 21 November sample; a second annotator is being sought for a blind 50-pair overlap, and if none is found before unblinding the single-annotator limitation is stated | No |
| 6 | Second encoder | In scope: BGE-base-en-v1.5, revision pinned in `E4_SECOND_ENCODER.md` before encoding; margins ±0.01 | No |

Before freezing, `docs/protocol/E4_SECOND_ENCODER.md` must exist with the pinned revision.

## 6. Freezing

When no open decision remains, rename this file to `EXTENSION_PROTOCOL_V1.md` and run:

```bash
python -m research.longeval.freeze_extension BUILD_PARENT    # the folder that holds the study's build/ outputs
```

It writes a receipt with the time, the commit and the SHA-256 of this protocol, the thresholds
record, the judge, coverage, audit, evaluation and insertion-order code, their job scripts and
runbooks, and the pinned model and package lists. It refuses if any of those files has an
uncommitted change. It refuses to run while the protocol holds an open decision, and it
records which follow-up outcomes already existed at that moment.
