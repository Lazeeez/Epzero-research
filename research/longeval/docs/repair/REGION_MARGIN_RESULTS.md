# Regional progress-bound feasibility result — 2026-11-18

**Decision: do not promote this isotropic-ball certificate to a retrieval experiment.**
The fit-only gate in [REGION_MARGIN_PLAN.md](REGION_MARGIN_PLAN.md) was written
before execution. None of the available proposals passed in any of the three
frozen fitting splits. This is a failed candidate feasibility test, not a failed
native method benchmark and not evidence that all regional repair is ineffective.

| Split | Fitting queries | Eligible proposals | Positive progress at generating query | Positive bound over chosen ball |
| --- | ---: | ---: | ---: | ---: |
| 0 | 30 | 21 | 4 | 0 |
| 1 | 30 | 17 | 3 | 0 |
| 2 | 30 | 20 | 3 | 0 |

These are58 proposal/split occurrences from overlapping subsets of the same
previously recorded32 proposals, not58 independent edges. Each radius is the
distance to the nearest other fitting query. Even the largest ratio of certified
radius to that distance is only0.04558; split maxima are0.04558,0.01029 and0.04329.
The median ratio is zero in every split because most edges fail source-relative
positive progress even at their generating query.

This exposes a mismatch in the proposed objective: a source may already be a
closer neighbor than an additional missing top-100 result. A useful coverage edge
therefore need not improve on the source's score. Our inference is that requiring
source-relative monotonic progress would discard most of these candidate coverage
edges. It does not establish a theorem about all graph repair or all query regions.
The ball bound is also conservative; failing a sufficient bound does not demonstrate
failure on an actual neighboring query. A test explicitly demonstrates that distinction.

The next method should target additional result coverage per unit cost rather
than assume source-relative score improvement is necessary. Simply making this
bound less strict, adding query perturbations or using approximate references is
not an established distinction from prior graph construction/repair methods.
No new method has passed the promotion gate.

Seven mathematical/input tests pass, including attainment at the worst point of
the ball and a counterexample to interpreting certificate failure as transfer
failure. Input hashes match the frozen sample and witness proposal report. The
audit accesses no tuning/development vectors in its computation and performs no
new retrieval evaluation. All381 held-out outcomes remain untouched.

Code: `region_margin.py`; execution driver: `build/run-longeval-region-margin-01.py`.
Evidence: `build/longeval-region-margin-01/report.json`, with all58 per-proposal
records, the frozen protocol and source/input/plan hashes. No PACE job is needed
for this rejected rule.
