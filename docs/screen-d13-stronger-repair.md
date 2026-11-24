# Development screen D13: a stronger repair (8 links and refining new insertions)

Recorded on 24 November 2026, before it was run, after the full-scale engine arm missed the
1.0-point target on the test queries (repaired excess 1.54 points, `scripts/pace/LONGEVAL_ENGINE_ARM.md`).

## Design

As D10 (product path on the 300K development sample: 8-bit codes, `fast_build` on, M 16,
efConstruction 100; snapshot 1 bulk-built and refined; snapshots 2 and 3 inserted one at a time,
each snapshot's insertions refined with `refine_links(threads=all, ef=50, first=...)`), with
`extra_backlinks` 8 instead of 4, seeds 11, 29 and 47. Compared with the grown, repaired (4 links)
and rebuilt arms of D7, D9 and D10 on the same seeds. Measures as D10: the newest cohort's
within-query excess, the oldest cohort's miss rate and recall@100 at ef 200 on the test queries, plus
insertion and refine time.

## Prediction

Averaged over the three seeds, 8 links cut the excess below the 4-link repair's 0.63 points to at most
0.40 points, raise the oldest cohort's miss rate against grown by at most 0.4 points (D7b, without
the refine of new insertions, gave +0.31 with 8 links), and leave recall@100 no lower than the 4-link
repair. If the oldest cohort rises by more than 0.4 points, 8 links do not go to a full-scale run.

## Result

(Run 24 November 2026, after the design above was committed.) Ef 200, test queries of the 300K sample;
grown, 4-link and rebuilt arms from screens D7, D9 and D10 on the same seeds. Raw:
`Oryx/build/screen-d13/*.json`.

| Arm | Newest-cohort excess (seeds 11 / 29 / 47; mean) | Oldest-cohort miss, mean | Recall@100, mean | Distance computations |
| --- | --- | --- | --- | --- |
| grown | 2.27 / 2.10 / 2.30; **2.22** | 4.09% | 0.9475 | 3,033 |
| 4 links + refine new (D9, D10) | 0.53 / 0.57 / 0.78; **0.63** | 4.32% | 0.9509 | 3,108 |
| **8 links + refine new** | 0.28 / 0.13 / 0.43; **0.28** | 4.45% | 0.9506 | 3,107 |
| rebuilt | 0.22 / −0.05 / 0.21; **0.12** | 4.46% | 0.9520 | 3,156 |

**Predictions:**
- "Mean excess at most 0.40 points, below the 4-link repair's 0.63": **held** (0.28; −88% against grown;
  every seed below the 4-link repair's).
- "Oldest cohort's miss rate at most 0.4 points above grown": **held on the mean** (+0.36), not on every
  seed (+0.34 / +0.31 / +0.43).
- "Recall@100 no lower than the 4-link repair's": **failed narrowly** (0.9506 against 0.9509; the
  seeds differ by up to 0.0011).

**Reading:** 8 links bring the grown index to within 0.15 points of a rebuild's excess, with the same
search work as 4 links. The oldest cohort's rise is the same as a full rebuild's (4.45% against 4.46%):
it is what removing the bias costs that cohort, the losses moving back from new to old documents
(section 4.6b of the research document), not damage peculiar to the repair. Insertion took 72 s on each
seed (4 links: 69–88 s; laptop timings, not comparable to within 10%), refine 8 s on all cores.

**Decision:** under the rule above (mean rise 0.36 ≤ 0.4), 8 links go to a full-scale run as a fourth
engine arm; decision 3's margins (excess ≤ 1.0 point with an interval excluding 0; oldest ≤ 0.75; recall
≤ 0.5) are the test there.
