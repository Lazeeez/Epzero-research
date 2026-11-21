# Extension protocol (draft)

**Draft of 21 November 2026, for advisor review. Not frozen.** It follows revision 3 of the author's
follow-up plan. Points marked **TO DECIDE** are open; the protocol cannot be frozen while any remains
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
| Judge pilot on 270 prompt-development pairs | **TO DECIDE:** add the job number and the evidence checksum when the pilot has returned |

## 3. Experiments

### E1. Judgment coverage

**Assessor.** GPT-OSS-120B run locally with the organisers' prompt and answer parser (`judge.py`,
pinned by tests). The serving stack and model revision are those of the judge pilot
(`judge-requirements.txt`, `judge-model.json`).

- **TO DECIDE:** the sampling setting (provider default or temperature 0) and the answer cap, from
  the pilot's repeat agreement.
- **Validation.** As recorded in `JUDGE_VALIDATION_THRESHOLDS.md`, on 1,000 officially graded
  pairs that exclude the 270 pilot pairs. **TO DECIDE:** approval of the thresholds, or the
  stricter bar of its section 5.

**Pool.** The top 10 of exact search and of the contributing runs, at all seeds and snapshots,
graded once per distinct query–document pair, including pairs that already have an official
grade.

- **TO DECIDE:** the contributing runs. The primary arms give 5,511 pairs (445 never graded);
  adding the shuffled-order runs and the ef sweep gives 6,298 (1,138 never graded).

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
- **Panel.** The 374 queries of the frozen primary panel.

### E2. Insertion order at full scale

Design, datasets and definitions are fixed in `scripts/pace/LONGEVAL_INSERTION_ORDER.md`: both
implementations, three seeds, three insertion orders, on all of snapshot 3 and on three
stratified draws of snapshot-1 size.

- **Primary order effect.** The as-arrived miss ratio minus the reversed miss ratio at ef 200, for
  snapshot-3 arrivals against snapshot-1 arrivals, on the full snapshot: per implementation, as
  the mean over three seeds, with a paired query bootstrap.
- **TO DECIDE:** which contrast counts as the reversal test. The pilot shows a reversal between
  the two extreme arrival groups and only a reduction when snapshots 2 and 3 are pooled, because
  the snapshot-2 block is in the middle of both orders. Recommended: snapshot 3 against
  snapshot 1; both are reported either way.
- **Secondary.** The same contrast at ef 100, 400 and 800; the decile curve of the shuffled
  build; incoming links by insertion decile; the three draws, with draw variation reported
  separately from seed variation.

### E3. Human audit

200 pairs, blind to role: 100 documents dropped from an exact top 10 and 100 documents put in
their place, drawn over the three primary arms, their seeds and the three snapshots (`audit.py`).
Pairs that play both roles are excluded. Grades use the 0–2 scale of the organisers' prompt.

- **Outcome.** The mean human grade by role and their difference, with a query-clustered
  bootstrap interval.
- **Also reported.** Agreement with the official grades where one exists, and agreement between
  two annotators on a 50-pair overlap. **TO DECIDE:** who the second annotator is, or that there
  is none.

### E4. Second encoder

**TO DECIDE:** whether it is in scope for the first submission. If it is, the model, its revision
and text preparation, and the practical margin for the paired contrast between encoders must be
fixed here before any of its outcomes exists. If it is not, the paper states that the
effectiveness results rest on one representation, and that only the insertion-position effect was
replicated elsewhere.

## 4. Inference

- Seeds are averaged within each query before resampling.
- Intervals are 95% paired query-bootstrap intervals with 10,000 draws, as in the frozen
  analysis, unless a section above states otherwise. They are nominal per contrast.
- No contrast of this extension is a confirmatory test with an error rate controlled across the
  extension. If the second encoder is in scope, its three within-encoder arm tests form one Holm
  family at 0.05, with the frozen ±0.01 margin.
- A smaller, larger, reversed or imprecise effect is a valid outcome and is reported as such.

## 5. Open decisions

Every **TO DECIDE** above, collected:

1. Sampling setting and answer cap of the judge (after the pilot).
2. Approval of the judge thresholds.
3. Contributing runs of the judging pool.
4. The reversal test of E2.
5. The second annotator of E3.
6. Whether the second encoder is in scope, and if so its model and margin.
7. The judge pilot's job number and evidence checksum in section 2.

## 6. Freezing

When no open decision remains, rename this file to `EXTENSION_PROTOCOL_V1.md` and run:

```bash
python -m research.longeval.freeze_extension BUILD_PARENT    # the folder that holds the study's build/ outputs
```

It writes a receipt with the time, the commit and the SHA-256 of this protocol, the thresholds
record, the judge, audit and insertion-order code, their job scripts and runbooks, and the pinned
model and package lists. It refuses to run while the protocol holds an open decision, and it
records which follow-up outcomes already existed at that moment.
