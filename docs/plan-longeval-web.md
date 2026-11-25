# Plan: a second longitudinal collection (LongEval Web 2025)

Written on 26 November 2026. A plan, not a design record: nothing here has been downloaded or run, and the
design and predictions will be recorded separately before any job.

## Why

Every effectiveness result of the paper comes from one collection that only grows. The arrival-bias results
were reproduced on public data with random cohorts and on the Big-ANN stream, but never on a second real
collection that changes over time, and never with documents that are removed or rewritten.

## The collection

LongEval 2025 Web Retrieval (TU Wien Research Data Repository, record th5h0-g5f51, also through the LongEval
`ir_datasets` integration): Qwant web pages and queries, mostly French with English translations, released as
monthly snapshots (training June 2022 – February 2023; test March – August 2023), about 124 GiB in five files,
with click-model relevance labels. Reported sizes differ between the lab's papers (14–18 million documents
per period, 33 million in all); the README is the authority. **Licence:** Qwant LongEval
Attribution-NonCommercial-ShareAlike, to be accepted by the author before download.

Unlike LongEval-Sci, each monthly snapshot is a crawl: documents appear, change and disappear between months.
That makes it the test of the two scopes the paper cannot cover, deletions and rewritten documents, as well
as a second collection.

## What would be measured

1. **Neighbour level (no labels needed; first).** Document identity across months from the collection's IDs
   (and content hashes for rewritten pages). An index maintained month by month (inserts for new or changed
   pages, tombstones for removed or changed ones) against a rebuild of each month; the within-query miss ratio
   of the newest month's documents against the oldest live month's, as arrived and with the months reversed;
   the repair (8 links, refine at 100). Builds and searches on Gold 6226 nodes as before.
2. **Effectiveness (click labels).** D across months for the calibrated arms, with the click-derived labels
   only: they are sparser and noisier than the Sci lab's LLM labels, so this part is a robustness reading, not a
   confirmation.

## Feasibility

- Vectors: no organiser vectors are known for the Web collection, so documents must be encoded. With the
  E4 encoder (BGE-base-en-v1.5 on the English translations) and the pinned E4 stack on one H200, the earlier BGE
  job's throughput suggests a few GPU hours per 10 million passages; a multilingual encoder on the French text
  is the alternative. The encoder choice is a design decision to record first.
- Storage: 124 GiB of downloads plus about 3 KiB per document of 768-dimension float32 vectors (about 45 GB for
  15 million documents): within the 300 GB scratch allowance, one period at a time.
- Order of work: licence and download (author), document-identity audit across months, a 1-million-document
  development sample for the neighbour-level design, then full months.

## Decisions needed from the author

The licence; whether to use the English translations (one encoder, as E4) or the French text (a multilingual
encoder); and the scratch space to reserve.
