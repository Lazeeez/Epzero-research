# Development screen D1: does *who* links to a late document matter at equal in-degree?

Recorded on 23 November 2026, before it was run. Development data only (the 300K LongEval sample and
the 381 test queries); it chooses between engine designs and is not a confirmatory result.

## Question

Late-inserted documents have fewer incoming links, and repairs that add links help only partly
(`arrival-bias-research.md`, 4.5–4.6). If adding the *same number* of incoming links helps more when
the links come from particular nodes, the fix must choose donors (design C); if the donor choice does
not matter, adding links is enough (design A).

## Design

- **Graph.** FAISS HNSW (its base-layer lists can be edited directly), M 16, efConstruction 100,
  inner product, one thread, seed 11, built in arrival order on the 300K sample, exactly as the
  pilot builds it.
- **Targets.** Every snapshot-3 document in the sample (61,792), chosen without any query.
- **Eligible donors of a target.** Its 32 exact nearest documents (inner product) that do not already
  link to it at the base layer.
- **Intervention.** Each target receives up to 4 new incoming base-layer links from eligible donors.
  Targets are processed in one fixed random order; a donor accepts at most 8 new links in total. A
  donor with a free slot fills it; otherwise it gives up its least similar neighbour that was not
  itself added by the intervention. The same targets, eligible sets, caps and eviction rule apply in
  every arm; only the order in which donors are tried differs:
  - **closest**: the most similar eligible donors first;
  - **uniform**: eligible donors in a random order;
  - **hub**: eligible donors with the highest current in-degree first (well-connected nodes, a proxy
    for nodes that searches pass through).
- **Baseline.** The unedited graph.
- **Measures**, at ef 100, 200 and 400 on the 381 test queries (exact top-100 targets): the newest
  cohort's within-query excess (rate difference, snapshot 3 against 1); the oldest cohort's miss
  rate; overall recall@100; distance computations per query; links added and displaced per arm;
  mean in-degree of targets.

## Prediction

All three arms reduce the newest cohort's excess relative to the baseline. **closest** and **hub**
reduce it more than **uniform** at equal links added, and **hub** at least as much as **closest**.
If the three arms are within about 0.2 points of each other (the standard error of a two-build
comparison), donor choice does not matter at this scale and design A is preferred.

One seed and one build: a screen, not an estimate. A design it favours is tested with three seeds and
then on the confirmation readings.

## Result (run 23 November 2026, after the design above was committed)

FAISS HNSW, 300K sample, seed 11, as arrived; every snapshot-3 document (61,792) given up to 4 new
incoming links (about 247,000 links per arm); 381 test queries, exact top-100.

| Arm | Newest-cohort excess, ef 100 / 200 / 400 (pp) | Oldest-cohort miss, ef 200 | Recall@100, ef 200 | Distance computations per query, ef 200 | Links displaced | Target in-degree |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (unedited) | 2.14 / 1.36 / 0.64 | 4.56% | 0.9460 | 2,889 | — | 15.7 |
| closest | 0.14 / 0.35 / 0.21 | 4.50% | 0.9485 | 2,942 | 28,835 | 19.5 |
| uniform | 0.30 / 0.39 / 0.12 | 4.48% | 0.9487 | 2,968 | 26,639 | 19.5 |
| hub | 0.00 / 0.42 / 0.08 | 4.50% | 0.9487 | 2,959 | 75,594 | 19.4 |

**Against the prediction:** the three arms differ by at most 0.07 points at ef 200 (0.28 at ef 100,
0.13 at ef 400), within the 0.2-point noise of a two-build comparison, and **closest** and **hub** did
not beat **uniform**. At this scale, *how many* incoming links a late document has matters; *which*
nearby documents supply them does not. Design A (guarantee incoming links) is preferred over design C
(choose donors). The hub arm displaced nearly three times as many existing links for no gain.

**What the intervention did:** at ef 200 the newest cohort's excess fell from 1.36 to 0.35–0.42
points; the oldest cohort was not harmed (4.56% → 4.48–4.50%); overall recall rose 0.25 points; work
rose about 2–3% (distance computations). Searching wider on the unedited graph is far costlier for the
same purpose: at ef 400 (79% more distance computations) the excess is still 0.64.

**Limits:** one seed, one implementation, development queries; a post-hoc edit of the final graph
using exact neighbours, which an engine would not compute; only the newest cohort was treated. The
next step is an online version inside Epzero's insertion (every insertion guarantees itself incoming
links), evaluated on whole builds in all three orders.
