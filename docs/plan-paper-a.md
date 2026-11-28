# Plan: paper A, arrival bias in incrementally built graph indexes

Recorded on 28 November 2026. The LongEval effectiveness results stay in paper B (CLEF 2027,
`research/longeval/paper/manuscript-clef2027.md`); paper A is the systems result on its own.

## The paper in one paragraph

A graph index grown by appending serves its newest vectors worst. Each later insertion links to existing
nodes, so a node's incoming links grow with the number of nodes inserted after it (exposure,
E_i ≈ k̄ ln(n/i)), and the newest nodes are reached least. The effect is a property of incremental graph
insertion itself: two HNSW implementations, an independent Vamana and DiskANN's own library on SIFT, on a real
growing collection (LongEval-Sci), the Big-ANN stream and public data with random cohorts; heavy deletion
with tombstones doubles it. Reversing the order of the same cohorts reverses who loses. Extra back-links at
insertion plus a refine of each batch remove it for about 9% more insertion work, except under heavy
deletion. Streaming benchmarks average recall over the collection and cannot see it.

## Venue and timeline

| Step | Date |
| --- | --- |
| Experiments A1-A5 below | December 2026 to early February 2027 |
| Draft complete, internal review | 15 February 2027 |
| PVLDB volume 20 abstract / paper (the last monthly round of VLDB 2027) | 25 February / 1 March 2027, 5 pm Pacific |
| arXiv preprint | at submission, once the venue's preprint and anonymity rules are checked |

Fallbacks, in order: the next PVLDB volume's monthly rounds; SIGMOD 2028 (rounds not yet announced; SIGMOD
2027's were January, April, July and October); SISAP 2027 (2026's deadline was 12 June). One venue at a
time. Epzero is presented as one HNSW implementation among others, as in the existing manuscripts.

## What reviewers will ask, and the experiment that answers it

**A1. A second real stream with real arrival times** (the main gap: LongEval is one collection, Big-ANN's
stream is synthetic in its order). arXiv abstracts in submission order: the arXiv ID encodes the month,
and arXiv's metadata (abstracts included) is CC0 and available through OAI-PMH without an account.
Every abstract to the end of 2026 (about 2.8 million), embedded once with BGE-base-en-v1.5 on a PACE GPU (the LongEval pipeline),
inserted month by month. Queries: held-out abstracts from the last months (new content) and from across
the range, kept out of the index. The VIBE arXiv set cannot be used: its source has text only, no IDs or
dates, and its row count does not match. Design and predictions are recorded in `screen-a1-arxiv-stream.md`
before any vector is embedded.

**A2. Against the published repairs, at equal insertion cost.** CleanANN's bridge edges (SPAA 2026) and
NGFix (run on LongEval before, `LONGEVAL_NGFIX.md`, but on an 8,192-vector sample and by overall recall only, so
its arrival-level comparison needs new builds). CleanANN's code is public (`github.com/SylviaZiyuZhang/CleANN`,
MIT, a fork of DiskANN with Python bindings, linked from the paper's §6.1; an earlier note here that it was not
public came from a search that missed it). Its own implementation is the baseline, at the paper's settings (R 64,
α 1.2, L 75, L_I 64) and at R 32 beside the other Vamana arms; its bridge rule is GuidedBridgeBuild (same-depth
pairs of the insertion's search tree at depths log₂|D| + 2 to + 4).
Measured by the newest cohort's within-query excess and by overall recall, as everywhere else.

**A3. The mechanism, derived.** A closed form for the incoming links from later nodes, s(i), and from
earlier ones, r(i), under random insertion with pruning, checked against screen M1's measured in-degree by
position. Related work: preferential attachment and the age-degree correlation in growing networks.

**A4. Why DiskANN shows it less.** Its 1.3R slack on neighbour lists and its frozen start point varied one
at a time in the independent driver (screen M3's open question).

**A5. A repair for heavy deletion.** Under the Big-ANN deletion runbook the repair closed a quarter of a
doubled gap, because tombstones keep routing searches for the old documents. Arms: compaction (rebuild the
live set's links), and refining the live neighbours of tombstoned nodes when they are deleted.

## Material already in hand

Order rerun and reversal (LongEval, two implementations, frozen), random-cohort controls on public data,
the Big-ANN stream (`BIGANN_STREAM.md`), screens M1 (exposure), M2 (deletions), M3 (Vamana and DiskANN), X1
on hnswlib, the repair screens D1-D17, analyses R1 and R2.
