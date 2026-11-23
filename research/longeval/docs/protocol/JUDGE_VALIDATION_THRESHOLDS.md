# Judge validation: compatibility thresholds

**Recorded 21 November 2026, before any local judge was run. Proposed by the author; pending
advisor approval.**

*Extended on 21 November 2026, still before any judge output: the comparison with the released
human grades was added to section 5, and point 5 to section 6. The thresholds are unchanged.*

This file fixes how a locally run relevance judge will be checked against the organisers' released
grades, and what follows if the check fails. It is written down now so that the thresholds cannot
be chosen after seeing the judge's output.

- **No judge output exists.** No local model has graded any LongEval query–document pair. The
  only judge-related GPU step so far was an allocation test that loaded no model.
- **It is not part of FINAL_PROTOCOL_V1.** That protocol is frozen and unchanged. This file
  belongs to the planned follow-up on judgment coverage, which is exploratory with respect to
  the frozen protocol.
- **Changes are appended, never edited in.** After the first judge output exists, any change to
  this file goes in the [amendment log](#amendment-log) with a date and a reason.

## 1. Reference labels

The reference is the released GPT-OSS-120B test judgments: 28,208 distinct query–document pairs
over 381 queries, graded 0–3. A pair has the same grade at every snapshot where it is graded, so
the unit of validation is the pair.

| Official grade | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| Distinct pairs | 9,716 | 5,552 | 12,936 | 4 |

If hardware forces the smaller model, the reference becomes the released GPT-OSS-20B labels, the
local judge is GPT-OSS-20B, and the same thresholds apply. The 120B frozen comparison then stays
a separate benchmark anchor. The reference family is fixed before the validation sample is graded.

## 2. Validation sample

- **Size and unit:** 1,000 distinct pairs that have an official grade.
- **Strata:** official grade × the first snapshot at which the pair is graded. All four grade-3
  pairs are included.
- **Drawn before the judge runs,** with a recorded seed. The sampling manifest stores each pair's
  stratum and selection probability. The manifest holds identifiers, so it is not published; its
  SHA-256 is.
- **Disjoint from prompt development.** Any pairs used to develop the prompt or the grade parser
  are drawn separately and excluded from the validation sample.
- **Blind.** The judge sees only the prepared query and document inputs. It does not see the
  official grade, the system, the seed, the rank or the snapshot.

## 3. Measures and thresholds

All three must hold for the local judge to be called compatible with the organisers' assessor.

| # | Measure | Threshold |
| --- | --- | --- |
| 1 | **Binary agreement:** share of pairs on which both assessors agree whether the grade is above 0 | at least 0.90 |
| 2 | **Linear-weighted Cohen's kappa** on the four grades 0–3 | at least 0.80 |
| 3 | **nDCG@10 agreement:** absolute difference between mean nDCG@10 under local grades and under official grades, on complete exact-search top-10 lists | at most 0.02, at every snapshot |

Definitions, fixed here:

- **Weights.** Measures 1 and 2 are population-weighted: each sampled pair is weighted by the
  inverse of its selection probability, so that over-sampled rare grades do not distort the
  overall figure. The unweighted confusion matrix and the agreement by official grade are
  reported beside them.
- **Decision.** Each threshold is compared with the point estimate. A 95% interval from a
  query-clustered bootstrap (queries resampled, 10,000 draws) is reported beside it. If the
  interval reaches below a threshold that the point estimate meets, the result is reported as
  "met, with an interval that includes the threshold".
- **Lists for measure 3.** A list is the exact-search top 10 of one query at one snapshot. It is
  complete when all ten documents have an official grade. The query sample is fixed in the
  sampling manifest before the judge runs. Both assessors grade the same ten documents, and
  the ideal DCG of each list is computed from those ten documents, with the same gain mapping
  (the grade itself, as in the frozen evaluation). A list whose ideal DCG is zero scores 0 and
  is counted.

## 4. What follows from the outcome

| Outcome | Reading |
| --- | --- |
| All three thresholds met | Local grades are described as compatible with the organisers' assessor. The "official plus filled" estimate may be reported as a supplement to the release. |
| Any threshold missed | Every estimate that uses local grades is reported as **sensitivity under the local assessor**. No statement is made about what the organisers' assessor would have graded. |

In both cases the primary coverage estimate is the same-assessor contrast (local grades with all
grades visible, against local grades with the official judgment gaps imposed). It uses one
assessor throughout, so it does not depend on this validation. The validation decides only how
local grades may be combined with, or described next to, the official ones.

**One configuration, one attempt.** The model revision, precision, serving runtime, prompt,
document fields and their order, truncation, decoding options, grade parser and gain mapping are
pinned before the validation sample is graded. If validation fails and the configuration is then
changed, that is a new attempt: it is recorded in the amendment log, and every attempt is
reported, including the failed ones.

## 5. Reference point: agreement between the two released label sets

The organisers released two label sets for the same pairs, from GPT-OSS-120B and GPT-OSS-20B.
Their agreement shows what a *different* model reaches on the same pipeline, and so how demanding
the thresholds above are. It was computed on 21 November 2026 from the released judgments only
(`heldout_pool.label_agreement`; output `build/longeval-heldout-pool-01/report.json`).

| Measure, 20B against 120B, all 28,208 pairs | Value | 95% interval (query-clustered) | Proposed threshold |
| --- | --- | --- | --- |
| Binary agreement | 0.932 | [0.923, 0.940] | ≥ 0.90 |
| Linear-weighted kappa | 0.835 | [0.821, 0.847] | ≥ 0.80 |
| Difference in mean nDCG@10, complete exact top-10 lists (362–366 per snapshot) | +0.004 / +0.000 / +0.003 at snapshots 1 / 2 / 3 | — | ≤ 0.02 |

**The released 20B labels would pass all three thresholds against the 120B labels.** The
thresholds therefore show that a local judge is no further from the official 120B grades than the
smaller model of the organisers is. They do not show that it reproduces the 120B assessor. Two
consequences:

- A local 120B judge that only just passes should be described as "as close as the released 20B
  labels", not as a reproduction.
- A stricter bar is available if the advisor wants one: require the local 120B judge to agree
  with the official 120B grades at least as well as the released 20B labels do (binary agreement
  0.932, kappa 0.835).

### Agreement of the released LLM grades with the released human grades

The release also holds human grades for 29 queries at snapshot 3 (1,581 pairs, scale 0–2). Against
them the official 120B grades reach a binary agreement of 0.70 [0.62, 0.78] and a linear-weighted
kappa of 0.24 [0.14, 0.36]; the model grades 87.0% of these pairs as relevant and the human grades
60.4% (`heldout_pool.human_agreement`; details in section 11 of HELDOUT_EXPLORATORY.md).

This sets the limit of what the validation above can show. Agreement with the official LLM grades
means that the local judge reproduces the organisers' assessor, including its leniency. It does
not mean that the grades agree with human judgment. The local judge's agreement with these human
grades will be reported beside the three measures, with no threshold, so that a reader can see
both.

## 6. Decisions (formerly open points for the advisor)

Decided by the author on 23 November 2026, the advisor being unavailable, before any local judge
grade of the validation sample or the pool exists.

1. **Thresholds:** the three of section 3 are the gate. The stricter bar of section 5 (binary
   agreement ≥ 0.932, kappa ≥ 0.835) is reported as a second tier: meeting it allows "at least as
   close to the official grades as the released 20B labels".
2. **Decision on the point estimate**, with the 95% query-clustered interval reported beside it.
3. **Measure 3 uses every complete exact-search top-10 list** at each snapshot (362–366 per
   snapshot), since the full pool is graded anyway.
4. **Reference family: GPT-OSS-120B**, the model run locally in the judge pilot.
5. **Agreement with the released human grades is reported without a threshold.**

## Amendment log

- 23 November 2026: section 6 decisions recorded (see above).
