# Method-paper research plan — 2026-11-18, exploratory v1

The primary objective is now a **new deployable ANN method with a competitive
quality–cost tradeoff**. The author explicitly requested this change. The existing
LongEval history/maintenance study supplies diagnosis, controls and evaluation
infrastructure; it is no longer the intended central contribution. A method paper
requires both a defensible distinction from prior methods and measured benefit.
Neither has been established yet.

**First screen completed:** see [FRONTIER_PILOT_RESULTS.md](FRONTIER_PILOT_RESULTS.md).
The initial overlap heuristic passed mechanics checks but failed the promotion gate
against wider search at equal actual distance work. Retain it as a negative result;
do not submit a large sweep of this version. The specification below records the
first candidate and defines requirements for the next justified iteration.

**Second screen completed:** [WITNESS_REPAIR_RESULTS.md](WITNESS_REPAIR_RESULTS.md)
records a calibration-only path-repair overlay. Its gains did not transfer meaningfully;
all 32 destinations are calibration exact neighbors with single-query support.
Do not promote this version. A new full-corpus coverage audit and a three-seed native
[competitor harness check](STANDARD_GRAPH_RESULTS.md) are complete. The next candidate
needs a structural signal that does not rely on memorizing these 50 calibration
queries, or a declared larger disjoint training-query population before learning.

**Mechanism audit completed:** [TRACE_PHASE_RESULTS.md](../development/TRACE_PHASE_RESULTS.md)
rechecks 1,800 full-data traces. Final development misses at the selected work
budget occur after some true neighbors are found. Neighborhood coverage is the
next target, but [NGFix/RFix](NGFIX_REPRODUCTION.md) already repairs that mechanism.
Its pinned native sample screen is now independently verified; see
[NGFIX_RESULTS.md](NGFIX_RESULTS.md). NGFix gains 0.46 percentage points on development
at ef 100 but 6.12 points on its fitting queries. Reusing those queries to choose ef
misses development recall targets; separate fitting and tuning roles are required.
The next candidate needs algorithm-level differentiation from this competitor,
not simply a new hardness score or a historical-query shortcut overlay.

**Protocol correction completed:** job 7148412 verifies all three 30/20/50 splits;
all 24 selected points meet development mean-recall targets. NGFix shows small gains
and similar latency, not a new method win. See [NGFIX_DISJOINT_RESULTS.md](NGFIX_DISJOINT_RESULTS.md).
The baseline rankings/counters reproduce, but raw saved-state hashes differ. Future
adapters must initialize delete flags and export canonical active-state fingerprints;
do not equate raw serialization with graph identity across independent builds.

**Additional feasibility rejection:** [REGION_MARGIN_RESULTS.md](REGION_MARGIN_RESULTS.md)
tests a sufficient source-relative progress bound across fitting-query regions.
None of 58 overlapping proposal/split cases passes. This objective discards most
existing coverage shortcuts even at their generating query. Do not promote the
isotropic-ball rule; it is not a distinct validated algorithm. Alpha-convergent
graphs and delta-EMG also constrain any future geometric-guarantee novelty claim;
see the primary-source review in [REGION_MARGIN_PLAN.md](REGION_MARGIN_PLAN.md).

## What the evidence permits

The two seed-11 failures are recoverable on the same graph. Hand-selecting an
evicted candidate helped, but all 48 equal-distance-work comparisons matched wider
search exactly. The 4,000-distance prefix improves robustness over ef=200 while
also changing work and beam width. Neither finding is a new efficient algorithm.
Ordinary query-adaptive stopping already exists in Epzero (`patience`, recall
guard) and published methods. Rebranding it is not the research objective.

## First falsifiable candidate: overlap-guided frontier resumption

Hypothesis: after a narrow beam exhausts, some of its nearest discarded candidates
lead predominantly back into the explored region. Cheap adjacency-ID information
may choose a less redundant escape route before spending expensive vector distance
calculations. The candidate changes **which node gets expanded**, not only when
search stops. This is an unproven heuristic, not a novelty claim.

The additive prototype is `frontier_search.py`:

1. Run the original upper-layer descent and ordinary distance-sorted layer-0 beam.
2. Retain a bounded distance-sorted reserve of scored, unexpanded candidates that
   were rejected or evicted. Maintain the same visited set and existing scores.
3. When the ordinary beam exhausts, inspect adjacency IDs for the nearest R reserve
   candidates. For candidate v compute `|N(v) minus visited| / |N(v)|`, with zero
   for an empty list. Select the largest fraction; break ties by distance and then
   discovery order. This does not inspect vectors or know exact neighbors.
4. Expand that candidate, resume the ordinary beam, and repeat within a distance
   budget and a resumption limit. Include all upper-layer work in the cap. Keep the
   index unchanged and report top-k over scored candidates retained by the beam.

Initial parameters, chosen before the small real-vector pilot: ef=200, reserve=256,
R=8, at most 8 resumptions, k=100, caps 2,000 and 4,000. The initial pilot uses
8,192 uniformly sampled snapshot-1 rows (NumPy seed 20261118), construction seed 11,
M=16, efConstruction=100, all 100 already-exposed training queries. Exact references
are computed **within this sample**, not borrowed from the full corpus. This pilot
checks execution and possible failure modes; sampling can remove the actual traps.
It cannot qualify a method for publication or justify a full-corpus speed claim.

The API accepts no query IDs, judgments, exact neighbors or designated bridge IDs.
It explicitly counts lookahead adjacency reads/IDs, expansions and distance calls.
Unvisited fraction can favor an irrelevant branch; a test demonstrates this
counterexample as well as a favorable example. The reserve can lose useful routes.
The Python implementation copies vectors through the diagnostic interface and is
unsuitable for comparing wall-clock query speed with native competitors.

## Required ablations and promotion gates

First compare against the same graph with: ordinary search; wider ef with the same
actual distance work; closest-candidate resumption using an identical reserve;
deterministic random reserve selection; existing patience; and restart-and-double-ef
with all repeated work charged. R=1 removes the overlap choice. Vary reserve size
and resumption limit on calibration only. Include adjacency/memory overhead, not
just distance savings. Do not use only the two selected failure queries.

1. Correctness: off-mode full native trace/ranking/counter parity; hard budget
   enforcement; independent score checks; immutable graph; no truth-dependent input.
   Ten initial tests pass, including 12 native query/effort/dimension parity cases.
2. Full development: all three seeds/snapshots and both training roles, with recorded
   calibration choices. Reject a policy whose apparent benefit is merely extra work
   or that is dominated by closest resumption/wider search. The sampled pilot alone
   cannot establish this. Preserve negative results and avoid unlimited heuristic
   sweeps on this already-small, repeatedly exposed query collection.
3. Native implementation: only promote a promising traversal to a versioned C++
   implementation. Re-run parity, actual latency, allocation and memory checks.
   The old 127-file experimental implementation and its bundles remain reproducible.
4. Competitor screening: an initial engineering target is at least 10% lower native
   query time at approximately matched achieved ANN recall (within 0.2 percentage
   points), without a material failure-tail regression. Use .95 and .99 target curves;
   do not infer a win from a target miss or select an advantageous operating point.
   These are screening targets, not final statistical acceptance thresholds.
5. Confirmatory evaluation: freeze the method, competitor builds/configuration
   selection, practical margins and inference before any of the 381 held-out query
   retrieval outcomes are inspected. Add at least two external settings with different
   dimensions/distributions for a general method claim. Obtain larger independent
   query sets for stable tail estimates. Preserve a failure result if the idea fails.

If routing changes repeatedly fail at equal work, reconsider the mechanism (for
example a separately specified repair method). Do not silently turn adaptive
parameter tuning or an empirical null result into the promised new method.

## Competitor performance is a primary deliverable

| Comparison | Why needed | Current state / next action |
| --- | --- | --- |
| Epzero original, wide beam, patience, closest/random resume | Isolate the new rule from extra effort and candidate reuse | Original/replay validated; closest and overlap prototypes implemented; remaining ablations pending |
| FAISS HNSW and hnswlib | Strong standard graph implementations | Full native Epzero/FAISS timings verified at original/calibrated/wide settings; FAISS lower final means at common ef; hnswlib 0.8.0 three-seed sample comparison validated; tuned full native curves remain needed |
| DARTH and Ada-ef | Direct competitors for query-dependent work and recall control | Primary papers/artifacts located; reproduce paper-matched pinned versions before comparison |
| In-memory DiskANN/Vamana | Alternative graph construction/search family | Select and pin a compatible CPU/IP configuration; include construction and memory budgets |
| NGFix/RFix | Direct published neighborhood/route-repair competitor | Job 7148412 disjoint-role sample verified: all 24 target points met, modest recall gains at similar latency. Canonical state diagnostics, explicit flag initialization, full-data and broader training remain |
| FAISS IVF-Flat, frozen and refreshed | Different search/maintenance mechanism | Three-seed full development controls verified; native cost measured |
| Quake; RoarGraph/adaptive entry selection | Relevant if retaining maintenance or OOD-routing claims | Related work now; benchmark when the actual claim overlaps, with required training resources disclosed |

Use the same vectors, metric, exact references, k, query population, hardware and
thread count. Choose configurations from calibration Pareto curves under declared
index-memory/build budgets; a common ef or M does not imply common cost. Include
disjoint fitting and parameter-selection roles for learned or repaired indexes;
never certify a target using only the queries used to fit the graph. Include
training, calibration, insertion/repair and refresh costs and amortization. A learned
competitor must receive appropriate training resources; starving it with only 50
queries is not a fair demonstration of superiority. Any additional training queries
must be disjoint from the held-out test and disclosed for every method.

Primary performance plots should show achieved recall versus native latency/QPS,
including mean and per-query p50/p95, plus low-recall query rates and worst-decile
recall. Report relevance nDCG and relevant-document recall separately. Fifty-query
p99 estimates are unstable; repeated timing does not create new independent queries.
Use paired query-level analysis, retain every seed, and report failed target points.
Only within-engine audited counters are directly comparable; FAISS and Epzero
distance-count conventions differ. Python replay latency is not a competitor result.

The `longeval-pace-native-cost-02.tar.gz` baseline jobs 7139467/8/9 have returned and
passed independent verification; see [NATIVE_COST_RESULTS.md](../development/NATIVE_COST_RESULTS.md).
They do not evaluate a new method. Epzero's lower time at its selected ef 200 versus
FAISS's selected ef 400 for seeds 29/47 accompanies lower achieved recall. The wide
ef 3200 comparison favors FAISS mean latency at similar mean recall. Neither
comparison supplies a new-method win or a tuned minimum-cost99% operating point.
All large PACE inputs/work files must remain on compute-node temporary disk.

## Prior-art boundary — initial primary-source review, not exhaustive

| Source | Established overlap / implication for this project |
| --- | --- |
| [DARTH](https://arxiv.org/abs/2505.19001), [artifact](https://github.com/MChatzakis/DARTH) | Learned adaptive early termination and requested recall already exist. Pin the publication's implementation; repository HEAD now describes extensions and is not automatically the paper version. |
| [Ada-ef](https://arxiv.org/abs/2512.06636), [artifact](https://github.com/chaozhang-cs/hnsw-ada-ef) | Distribution-aware query-adaptive ef is prior work; a new ef predictor alone is insufficient differentiation. |
| [Adaptive entry selection](https://arxiv.org/abs/2402.04713) | Query-dependent starts are already studied; adding a different start is not by itself novel. |
| [RoarGraph](https://arxiv.org/abs/2408.08933) | Query-distribution-guided graph construction is prior work, particularly for OOD/cross-modal search. |
| [Quake](https://arxiv.org/abs/2506.03437) | Adaptive vector-index maintenance is prior work; temporal evaluation alone does not create a new maintenance method. |
| [NGFix/RFix](https://arxiv.org/html/2510.22316v1), [pinned artifact](https://github.com/yuhuifishash/NGFix/tree/9acd987cf1c420069f8fd602c0d4e4aa81053558) | Escape hardness, historical-query local connectivity repair, route repair and insertion maintenance already exist. A direct competitor for the current mechanism, not peripheral related work. |
| [Navigation-aware DiskANN study](https://www.ijert.org/making-diskann-adaptive-navigation-aware-search-diversity-pruning-and-feedback-driven-graph-refinement-for-graph-based-ann-ijertv15is060328) | Reports usefulness-guided traversal and graph refinement. Its relationship to the proposed query-local adjacency rule needs algorithm-level checking; no claim of being first to use navigation feedback. |

The initial, now unsuccessful distinction investigated was **bounded query-local resumption chosen
using unscored adjacency overlap, with explicit accounting for that lookahead**.
Its pilot failed the equal-work gate. The current neighborhood-repair direction
still lacks a validated novel mechanism. Check graph diversification, corpus-proxy
construction, reverse links and query-guided repair at algorithm level before
implementing another candidate. The present search does not certify novelty.

## Artifact boundary

Before changing the objective, all 35 files in the verified IVF analysis manifest
and that manifest were archived and reverified in
`build/longeval-pre-method-20261118.tar.gz`, SHA-256
`26daf9b76d7dc9e07575a39843b3c8a6a823bc4dfb209832eabf50a9599930c1`.
The empirical manuscript remains supporting evidence, not a completed method paper.
