# First method-candidate screen — 2026-11-18

**Decision: do not promote overlap-guided frontier resumption v1 to a large PACE
sweep or a native performance implementation on the present evidence.** The small
real-vector pilot is correct and deployable in the sense that its decisions require
no truth labels, but it does not demonstrate an advantage over wider search at
equal actual distance work. This is a useful negative screening result, not a
full-corpus impossibility result or a publishable new method.

## Scope and verification

- 8,192 uniformly sampled snapshot-1 document vectors, dimension 2,560; sample seed
  20261118, construction seed 11, M=16, efConstruction=100, k=100, one thread.
- All 100 existing training queries, retaining the 50/50 calibration/development
  roles. The 381 held-out query retrieval outcomes remain untouched.
- Native baseline parity precedes intervention for every query. All returned scores
  are checked independently against float64 query/document products. Before/after
  serialized index hashes match. The entire corpus was not rehashed for this pilot;
  sampled rows/vectors and validated query-encoding identities are recorded.
- 800 search conditions: ordinary ef 200, closest reserve resumption, overlap-guided
  resumption, and ef 3200, each with caps 2,000 and 4,000. The method uses reserve 256,
  shortlist 8, and at most eight resumptions. These parameters preceded the pilot.
- An independent analysis recounts all 800 conditions and adds 400 paired widened
  controls, each capped to the candidate's **actual** distance evaluations. The old
  replay and new disabled-policy implementation agree on every distinct such run.
- Ten tests cover native parity, ties, caps, reserve bounds, hidden-neighbor recovery,
  a counterexample, and immutability. These tests establish mechanics, not efficacy.

## Equal actual work

For the overlap policy at the 4,000 cap:

| Population | Overlap mean recall | Wider control at identical actual work | Query wins / ties / losses |
| --- | ---: | ---: | ---: |
| Calibration, 50 queries | 98.06% | 98.22% | 0 / 44 / 6 |
| Development, 50 queries | 97.80% | 97.84% | 3 / 42 / 5 |

At the 2,000 cap all 100 overlap comparisons tie in recall. The simpler closest
resumption ties wider search in recall in all 200 paired conditions. Across the
200 overlap comparisons there are three wins, 186 ties and 11 losses. These are
descriptive exploratory counts, not an inferential claim. They do not establish
ranking equality when recall ties.

At the 4,000 cap the development cohort actually averages 2,246.48 distance calls
for overlap and 2,221.74 for closest resumption. Overlap inspects 5,492.64 adjacency
IDs on average versus 4,372.52 for closest, approximately 25.6% more. The 0.08 pp
recall improvement over closest uses more work; at equal actual distance work,
overlap's mean is below the widened control. Full ef 3200 at 4,000 calls obtains
99.60% mean recall, but spends substantially more distance work; that comparison
alone is not evidence against the candidate's efficiency.

## Consequence for the method paper

The first candidate fails the current promotion gate. Do not market its small gain
over ordinary ef 200 as algorithmic superiority, or treat Python timings as native
speed. Keep the candidate and its negative controls reproducible. The next method
iteration needs a stronger reason to select a useful escape path than the fraction
of unvisited neighbors; alternatively, choose a different mechanism after examining
the actual full-graph bottleneck and relevant prior art. Do not run an unrestricted
parameter sweep on the already-exposed training population.

The sampled graph can omit the original full-corpus traps. The result therefore
does not rule out all frontier-based methods. It also supplies no evidence that
this policy will beat FAISS, hnswlib, DiskANN, DARTH or Ada-ef. The primary objective
remains a new method plus measured competitive performance, as specified in
[METHOD_RESEARCH_PLAN.md](METHOD_RESEARCH_PLAN.md).

Artifacts: `build/longeval-frontier-pilot-01/` contains configuration, per-query
results, sample rows, exact references and unchanged sample indexes.
`build/longeval-frontier-comparison-01/` contains the independently recounted
analysis and equal-work pairs. `build/longeval-frontier-tests-01.xml` records tests.
