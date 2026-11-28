# Screen A1: arrival bias on arXiv in submission order

Recorded on 28 November 2026, while the metadata was being harvested and before any vector was embedded,
any index built or any result seen. For paper A (`plan-paper-a.md`). Exploratory replication on public data.

## Why

Every real-order result so far comes from one collection (LongEval-Sci). The Big-ANN stream has a real
runbook but synthetic, cluster-blocked arrivals, and the random-cohort controls have no real order at all.
arXiv is a second real collection that grows by appending, with each paper's arrival date public, and its
topics drift over time (new fields grow, old ones slow), so new papers land both among old ones and in
regions of their own.

## Data

- **Metadata:** every record of arXiv's OAI-PMH interface (`metadataPrefix=arXiv`; CC0) harvested by
  `research/arxiv_stream/harvest.py` from 28 November 2026: arXiv ID, first version's date, categories,
  abstract. Records without an abstract are dropped; duplicate IDs keep the first. Stored outside the
  repository; only aggregates are committed.
- **Order:** by the first version's date (`created`), ties by arXiv ID. Cohorts are the oldest, middle and
  newest thirds of that order by count.
- **Queries:** 10,000 abstracts held out of the index, fixed by seed 2027 before embedding: 5,000 drawn
  from the last 12 months of the harvest (**new-content queries**) and 5,000 uniformly from the whole range
  (**uniform queries**). Each set is reported on its own.
- **Vectors:** BGE-base-en-v1.5 (768 dimensions, CLS pooling, normalized; as the LongEval BGE arm, no
  instruction prefix, at most 512 tokens), embedded once on a PACE GPU. The job checks that re-embedding
  100 fixed abstracts reproduces their vectors to cosine 0.9999.
- **Exact targets:** inner product, exact float32 top 200 re-scored in float64, ties by position; the top
  100 is the target set.

## Arms

As the Big-ANN stream study (`BIGANN_STREAM.md`, `research/longeval/insertion_pilot.py run`, unchanged
where it applies): Epzero 0.6.0 without compression and FAISS HNSW, M 16, efConstruction 100,
single-threaded, one vector at a time. Orders `appended` (submission order), `reversed` (the newest third,
then the middle, then the oldest; submission order within each) and `shuffled` (one random permutation).
Seeds 11, 29, 47. Plus `repaired` (Epzero, appended order, `extra_backlinks` 8 and `refine_links(ef=100,
first=…)` after each calendar month's insertions). Searched at ef 100, 200, 400 and 800, k 100.

## Measures (ef 200; seeds averaged within query; query bootstrap, 5,000 draws)

1. **Primary:** in `appended`, the newest third's within-query excess miss rate over the oldest third's
   (Mantel-Haenszel, strata query × exact-rank band), and its paired change from `appended` to `reversed`.
2. **Position without content:** in `shuffled`, the within-query difference between the last and the first
   insertion tenth.
3. **Age curve:** in `appended`, the within-query miss rate by arrival year, relative to the overall rate.
4. **Repair:** `repaired` minus `appended` in measure 1; the oldest third's miss rate; overall recall@100.
5. **Exposure:** in `appended`, mean layer-0 in-degree by insertion-position decile against
   a + b·ln(n/i) (screen M1's model).

## Predictions

1. In `appended`, the newest third's excess is above +1 point with its interval excluding 0, for both
   implementations and both query sets.
2. Moving to `reversed` changes it by more than −1 point with the interval excluding 0 (the bias follows
   position, as on LongEval), for both implementations.
3. In `shuffled`, measure 2 is positive with its interval excluding 0 for both implementations.
4. The repair removes at least half of `appended`'s excess on both query sets, the oldest third's miss rate
   rises by at most 0.75 points (upper 95% bound), and recall@100 does not fall.
5. The exposure model fits the decile means with R² of at least 0.9.

A result against a prediction is reported as such. Aggregates only, in `research/arxiv_stream/`.
