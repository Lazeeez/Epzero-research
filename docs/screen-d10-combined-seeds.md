# Development screen D10: the combined repair over three seeds

Recorded on 24 November 2026, before it was run, after screen D9 (seed 11).

## Design

As D9 (product path: 8-bit codes, `fast_build` on, M 16, efConstruction 100; snapshot 1 bulk-built
and refined; snapshots 2 and 3 inserted one at a time), seeds 29 and 47, three arms each: grown with
no repair; grown with `extra_backlinks` 4 and each snapshot's insertions refined; rebuilt (reference).
With D7/D9's seed 11 this gives three seeds per arm. Measures as D9.

## Prediction

On each seed, the combined repair cuts the grown newest-cohort excess at ef 200 by at least half,
raises recall@100, and raises the oldest cohort's miss rate by at most 0.3 points; averaged over the
three seeds, the repaired excess is within 0.5 points of the rebuilt one.

## Result (run 24 November 2026, after the design above was committed)

Ef 200; seed 11 from screens D7 and D9.

| Seed | Grown excess | Repaired excess (cut) | Rebuilt excess | Oldest-cohort miss, grown → repaired | Recall@100, grown → repaired (rebuilt) |
| --- | --- | --- | --- | --- | --- |
| 11 | 2.27 pp | 0.53 pp (−77%) | 0.22 pp | 4.06% → 4.35% (+0.29) | 0.9476 → 0.9509 (0.9522) |
| 29 | 2.10 pp | 0.57 pp (−73%) | −0.05 pp | 4.18% → 4.37% (+0.20) | 0.9470 → 0.9503 (0.9521) |
| 47 | 2.30 pp | 0.78 pp (−66%) | 0.21 pp | 4.03% → 4.23% (+0.20) | 0.9479 → 0.9514 (0.9519) |
| Mean | 2.22 pp | 0.63 pp (−72%) | 0.12 pp | +0.23 | +0.0034 (0.9520) |

**Held on every seed:** the excess at least halves (−66% to −77%), recall@100 rises (+0.33 to +0.36
points), and the oldest cohort rises at most 0.3 points (+0.20 to +0.29).

**Failed narrowly:** averaged over the seeds, the repaired excess (0.63) is 0.50 points above the
rebuilt one (0.12); the prediction was "within 0.5", and the difference is 0.502. The repair closes
about three quarters of the gap to a rebuild, not all of it.
