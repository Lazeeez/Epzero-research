# Encoder-synthesized queries for query-aware graph repair — plan v1, 2026-11-18

Status: **executed; rejected at gate 4 (see PSEUDO_QUERY_RESULTS.md).** Original text below unchanged: This is the third method direction after the
frontier-resumption and witness-repair screens failed their gates. It is a
hypothesis with a measured motivation, not a validated method or a novelty claim.

## Motivation (measured on the local 8,192-document sample)

Real training queries are moderately out of distribution for the document vectors:

| Measure | Query → document | Document → document |
| --- | ---: | ---: |
| Mean top-1 inner product | 0.492 | 0.538 |
| Mean 100th inner product | 0.285 | 0.367 |
| PCA-128 energy captured (document basis) | 0.464 | 0.517 |
| Components of exact top-100 in exact document 16-NN graph | 6.6 | 3.8 |
| Isolated top-100 answers (no answer neighbor in that graph) | 4.9 | 2.5 |

The cosine between the mean query and mean document directions is 0.789. A query's
answer set is about twice as fragmented in document-neighbor structure as a
document's own neighborhood. The full-data trace audit found that all 523 remaining
misses at the selected work budget are *unencountered* neighbors of queries that
already found some answers. Fragmentation is a plausible structural cause.
This sample is easier than the full corpus; the effect size there is unmeasured.

## Hypothesis

Query-aware construction (RoarGraph) and query-driven repair (NGFix/RFix) connect
answer sets of real historical queries. Their limits here: they need a query log,
the log is small (100 exposed queries), and NGFix's gains did not transfer from fitted
to development queries in jobs 7141656/7148412. **The same encoder that produces
query vectors can produce in-distribution pseudo-queries from document text at index
time**, for every document, with no query log. If those vectors occupy the query
region, bridging their answer sets should repair the fragmentation real queries meet.

Candidate method (to be specified fully only after the feasibility gates):

1. At insertion, encode a short pseudo-query from each new document's title using the
   unchanged query instruction prompt.
2. Retrieve that pseudo-query's top-k with the current index (exact only in screens).
3. Add a bounded number of bridge edges between connected components of that answer
   set in layer 0, under a fixed per-node extra-degree budget.

Differences to test against prior work: no query log; repair is driven by the
inserted document itself, so maintenance is incremental under insert-only growth;
bridging targets answer-set components rather than all-pairs or hardness-ranked
edges. This is an engineering distinction until a deeper prior-art check confirms
that encoder-generated pseudo-queries have not been used for ANN graph construction.
An initial search (2026-11-18) found none; doc2query/synthetic-query work in IR
concerns document expansion and training data, not index topology.

## Stage 1 — PACE GPU encoding job (this package)

Inputs are prepared locally by `pseudo_queries.py prepare` and carried in the package.
Nothing about held-out query outcomes is used. Held-out and training query *texts* are
read only to drop pseudo-queries whose normalized text equals a real query; only the
count is recorded.

- **Set A (sample):** the 8,192 sample documents, two forms each (16,384 texts):
  `title` (whitespace-collapsed title) and `keywords4` (first four non-stopword
  lowercase title tokens; real training queries have median length 3, mean 3.9).
- **Set B (full corpus):** 131,072 snapshot-3 documents ordered by
  SHA-256(`longeval-pseudoquery-v1|` + ID), `keywords4` form only. For the later
  full-data screen; not analyzed until Set A passes.
- Prompt: the exact training-query instruction prompt. Pooling: last token.
  FP32 on A100 with TF32 disabled, matching the candidate query configuration.
  Batches contain only equal-length token sequences, so no padding is introduced.
- **Encoder gate (fail-closed):** the 100 real training queries are re-encoded through
  the batched path first and compared with the verified FP32 vectors from job 7107422.
  Every cosine must be at least 0.99999, or the job stops before encoding pseudo-queries.

## Stage 2 — local feasibility gates on Set A (decided before seeing results)

Development queries are used only for the final check in each gate; nothing is tuned on them.

1. **In-distribution:** pseudo-queries must be closer to real queries than documents
   are. Pass if the mean top-1 inner product from real queries to the pseudo-query set
   exceeds that to the document set, for at least one form.
2. **Coverage:** union of pseudo-query exact top-100 sets covers at least 90% of
   development queries' exact top-100 occurrences (the calibration-query witness
   coverage was 7.62% at full scale).
3. **Bridging potential (oracle-free):** adding pseudo-query bridge edges reduces the
   mean number of components of development answer sets in the augmented graph.
4. **Retrieval at equal work:** in the exact replay harness, the augmented graph
   improves development recall at equal distance computations over the base graph,
   and over NGFix fitted on 30 real queries, at ef 50/100/200.

A failure at any gate ends this direction with the result preserved. Passing all four
justifies a full-data CPU screen with Set B. Native implementation in Epzero and the
>=10% matched-recall latency gate in METHOD_RESEARCH_PLAN.md come after that.

## Risks stated in advance

- Titles may not resemble user queries; the keyword form may be too generic.
- Extra edges raise degree, and thus distance computations per expansion. Gains must
  survive equal-work comparison, not just equal ef.
- Encoding cost at index time (one 4B-model forward per document) must be reported as
  construction cost. A smaller encoder would change the representation and is a
  separate question.
- Positive sample results may not transfer to the full corpus.

The 381 held-out query retrieval outcomes remain untouched.
