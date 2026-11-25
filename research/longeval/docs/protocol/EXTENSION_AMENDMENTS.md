# Amendments to the extension protocol (version 1)

`EXTENSION_PROTOCOL_V1.md` and the files listed in its freeze receipt are not edited after the freeze
(23 November 2026). Changes of detail are recorded here, dated, with what was known at the time.

## 1. The arrival-order audit sampler (24 November 2026)

**What the frozen runbook says** (`scripts/pace/LONGEVAL_INSERTION_ORDER_IDS.md`, analysis D): from
"the as-arrived and reversed builds of seed 11 at ef 200", 50 dropped documents of snapshot 3, 50 of
snapshot 1, and the 100 put in their place, "sampled with seed 20261123 by a sampler added to
`audit.py` before the draw"; a second annotator grades "a 50-pair overlap".

**What is done instead, and why:**

- The sampler is a new module, `research/longeval/arrival_audit.py`, not an addition to `audit.py`:
  `audit.py` was frozen with the protocol, and editing it would break the freeze record. The new
  module reuses the frozen `audit.write_kit` unchanged, so the pages and key have the same form.
- "The builds of seed 11" is read as **both implementations' seed-11 builds, pooled** (Epzero and
  FAISS HNSW, as arrived and reversed), so that the audit is not tied to one engine. Each dropped pair
  gets one replacement from the same build and query.
- The second annotator's overlap is **52 pairs** (13 dropped pairs per arrival group with their
  replacements), the nearest balanced number to 50.

**Known at the time:** no output of the order rerun existed (its jobs had not been submitted), so no
audit pair could have been seen.

## 2. Fewer eligible pairs than the audit's 50 per group (24 November 2026)

**What happened:** with the order rerun's output in hand, the sampler (amendment 1) found only 46
dropped pairs from snapshot 3 in the seed-11 as-arrived and reversed builds at ef 200 (both
implementations, primary panel, excluding pairs seen in both roles), against the 50 per group the
runbook fixed. It stopped before drawing anything, as designed.

**What is done:** a balanced census at the smaller count. When either arrival group has fewer
eligible dropped pairs than 50, both groups take that many: all 46 dropped pairs of snapshot 3, 46
drawn from snapshot 1 by the same keyed hash, and one replacement for each; 184 pairs in all. The
second annotator's overlap stays at 13 dropped pairs per group with their replacements (52 pairs).
Pooling seeds 29 and 47 was rejected because it changes the population the runbook fixed.

**Known at the time:** the counts of eligible pairs by group (46 for snapshot 3); no pair had been
drawn, no page written, and no grade or official label of any eligible pair inspected.

## 3. Running E4 on the BGE vectors (25 November 2026)

**What the frozen protocol says** (section E4, `docs/protocol/E4_SECOND_ENCODER.md`): the held-out design
"with the BGE vectors in place of the released ones", and "the arrival-order reversal of E2 on BGE
vectors" (as arrived, shuffled, reversed; three seeds; M 16, efConstruction 100; ef 100–800; the full
snapshot 3).

**Why something has to be added:** every frozen module that reads vectors (`mapped.py`, `dense.py`,
`heldout.py`, `insertion_pilot.prepare_full`) accepts only the released Qwen archives, by their pinned
hashes, and frozen files are not edited.

**What is done for the arrival-order reversal:**

- A new module, `research/longeval/bge_sample.py`, writes the sample that `insertion_pilot run` reads,
  as `insertion_pilot.prepare_full` does: the same files, the same lexical order key, and exact
  neighbours by the same rule (float64 inner products, descending score, then descending document ID).
  Only the vectors differ: the verified BGE document and query vectors (job 7290861, files pinned by the
  SHA-256 its report records).
- The document order, each document's arrival snapshot and the query order are taken from the Qwen
  order rerun's saved sample metadata (job 7253605: `doc-ids.txt.gz`, `cohort.npy`, `query-ids.json`,
  `confirm-query-ids.json`), which `prepare_full` wrote from the released snapshots. The BGE document
  IDs must equal that order exactly, and the query ID sets must match.
- `prepare_full` checks its exact neighbours against the study's held-out references, which do not exist
  for BGE. In its place, the exact neighbours of a fixed subset of queries are recomputed by a full sort
  and must agree exactly.
- A test runs `prepare_full` (the frozen code) and the new module on the same stand-in data and requires
  identical outputs.
- Builds, searches and analyses are the frozen ones: `insertion_pilot run`, `insertion_strata` and
  `order_pairs`, unchanged.

**The held-out design on BGE** needs the frozen query-provenance chain (query records, representation
contracts, development references) rebuilt for BGE. It is recorded in its own amendment before it runs.

**Known at the time:** the BGE vectors and the checks in `LONGEVAL_BGE.md` (file hashes, shapes, unit
norms, document order, 22.1% of documents truncated, CPU re-encoding agreement). No BGE neighbour, exact
or approximate, had been computed, and no retrieval outcome on BGE existed.

## 4. The held-out design on the BGE vectors (25 November 2026)

**What the frozen protocol says** (section E4): "the held-out design of the frozen study (same queries,
panel, arms, seeds, calibration, labels), with the BGE vectors in place of the released ones", with the
frozen ±0.01 margins; its three within-encoder arm tests form one Holm family at 0.05.

**Why something has to be added:** the frozen loaders accept only the Qwen inputs. `test_queries.load_test_queries`
requires 2,560-dimensional vectors; `heldout.load_heldout` and `heldout.snapshot_specs` require the
released archives' hashes and the Qwen encoding receipts; `heldout_analysis.analyze_heldout` loads its
queries and references through `load_heldout`.

**What is done:** a new module, `research/longeval/bge_heldout.py`, supplies the inputs, and the frozen
cores do everything else, unchanged:

- **Snapshots.** Snapshots 1, 2 and 3 in the frozen reader's layout (`mapped.MappedSnapshot`: ingestion
  receipt, `pt_meta.json`, `docnos.npids`, `vecs.f4`), each holding the verified BGE vectors of exactly the
  documents of the released snapshot, in its row order (its `docnos.npids`, copied). The receipt carries
  a BGE identity in place of the release's archive hash.
- **Queries.** The calibration cohort is the frozen contract's calibration role
  (`training/queries/query-inputs.json`, read by the frozen `query_diagnostic.validate_inputs`); the
  evaluation cohort is the 381 test queries in the study's order. Their vectors are BGE's (job 7290861).
- **Exact references.** Computed per snapshot by the frozen `oracle.exact_mapped` (float64, descending
  score then ID) for both cohorts. At snapshot 3 the test queries' references must equal the exact
  neighbours `bge_sample.py` computed (amendment 3), a second code path.
- **Arms.** The frozen `trajectory.trajectory_core` (Epzero HNSW and FAISS HNSW; incremental and
  identical-history replay) and `ivf_trajectory.ivf_core` (frozen incremental, frozen rebuild, refreshed
  rebuild), with the frozen configurations (`trajectory-development.json`, `ivf-development.json`),
  seeds 11, 29 and 47, one thread, and the frozen provenance record (`heldout.heldout_provenance`), in the
  study's own environment (released Epzero 0.5.0): 15 jobs, as the frozen held-out run.
- **Analysis.** The frozen `heldout_analysis.collect` and `analyze_core` with the frozen labels
  (`load_test_qrels`) and panel; the BGE references in place of the Qwen ones.
- **Audits.** The frozen vector audit (which checks the release archives) is replaced by the BGE checks
  of `LONGEVAL_BGE.md` (hashes, shapes, finite values, unit norms, document order). The text audit is
  unchanged: the documents are the same.

**Tests:** the module's snapshot folders open in the frozen reader; its cohorts equal the frozen loader's
IDs on the Qwen inputs; its references on stand-in data equal a full sort; and a small end-to-end run
through `trajectory_core` completes.

**Known at the time:** the BGE vectors and their checks; the BGE order and engine-arm jobs (amendment 3)
had been submitted, and none had returned. No BGE retrieval outcome of any kind existed.

## 5. The order rerun's substitutions in the judging pool (25 November 2026)

**What the frozen protocol says** (section E1, decision 3): the pool is "primary arms plus the
shuffled-order runs and the ef sweep, 6,298 pairs (1,138 never graded), plus the top-10 substitutions of
the full-scale order rerun (E2b) on the test queries at ef 200, all three orders and seeds, added before
grading starts", each distinct pair graded once.

**Why something has to be decided:** the frozen `coverage.prepare` derives both the pairs to grade and the
scoring universe of the primary estimates from the same pooled sets of the ledger. The universe sets which
graded documents count for a query, including its ideal DCG, so adding the substitutions there would
change the primary arms' estimates at snapshot 3, which the protocol does not intend. The frozen
validation sampler also cannot draw an empty sample (it always keeps every grade-3 pair), so the
substitutions cannot be prepared on their own by `coverage.prepare` either.

**What is done:** two grading folders, graded in the same job under the same configuration.

- **Primary:** `coverage.prepare` unchanged, with the pooled sets `reported`, `+all_lists` and
  `+extension` of the judging-pool ledger (`longeval-heldout-pool-02`; 6,298 distinct pairs, as decision
  3 states) and its 1,000-pair validation sample. The frozen `coverage.analyze` reads it unchanged.
- **Substitutions:** the order rerun's top-10 substitutions at ef 200 (documents in a build's top 10 that
  are not in exact search's top 10; 2 implementations × 3 orders × 3 seeds, 381 test queries, snapshot 3):
  468 distinct pairs, of which 446 are already in the primary pool and graded there. The other 22 are
  written by a new module, `research/longeval/grading_inputs.py`, in the frozen input format (`pairs.jsonl`,
  `key.jsonl`, `universe.json`, `inputs.json`), which the frozen `coverage.grade` and `local_grades` read.
  Their grades serve the arrival-order analyses only; they enter no primary estimate.

**Known at the time:** the ledger, the order rerun's returned IDs and the count of substitutions. No
grade from the local judge existed for any study pair, and no grade of any substitution had been looked up.

## 6. Checking the BGE held-out inputs by content (25 November 2026)

**What amendment 4 says:** every held-out job prepares the BGE inputs itself (`bge_heldout prepare`), and the
analysis passes the frozen `heldout_analysis.collect` the hash of one job's `prepared.json` and of its gzipped
references, which collect requires every report to match.

**Why something has to be decided:** the 15 jobs (7298272-7298286, all passed) prepared identical inputs but
wrote 15 different `prepared.json` files: each records the temporary folder it wrote the snapshots to and the
hashes of its gzipped reference files, whose gzip headers carry the time they were written. Checked locally
before any analysis was run: the decompressed references of all three snapshots, the calibration and
evaluation query arrays, and every snapshot's archive and vector hashes are identical in all 15 jobs. The
byte-level check would stop the analysis at the second job, for a difference in nothing that a search reads.

**What is done:** `bge_heldout analyze` checks the inputs' identity on their content instead. For each report it
finds the job's own `prepared` folder, requires that the report's recorded hashes are that folder's, and requires
that folder's content fingerprint (decompressed references, query arrays and IDs, snapshot archive and vector
hashes, counts, the pinned BGE files and the sample-agreement check, without folder paths or gzip bytes) to equal
the analysis folder's. It then calls the frozen `collect` with no byte-level inputs check; every other check of
collect (configurations, completeness, duplicates, recounted recall) runs unchanged. `heldout_analysis.py` is not
edited.

**Known at the time:** that all 15 jobs passed, their evidence checksums, and the identity of their inputs as
above. No analysis of the BGE held-out runs had been run, and no D, miss rate or recall of them had been read.

## 7. Ten construction seeds on the BGE vectors (26 November 2026)

**What E4 says:** the BGE held-out arms use the frozen three seeds, 11, 29 and 47.

**Why something has to be decided:** with three seeds, the encoder contrast for Epzero's arm, +0.0075
[−0.0017, +0.0205], is not within the margin, and under Qwen ten seeds showed that one of the three (seed 29)
had pulled Epzero's D down (`HELDOUT_EXPLORATORY.md` section 18). Three seeds cannot separate construction
randomness from a difference between encoders.

**What is done:** an exploratory follow-up (`scripts/pace/LONGEVAL_BGE_SEEDS.md`) runs the seven further seeds
of the Qwen follow-up (101, 211, 307, 401, 503, 601, 701) through `bge_heldout.py`'s entry points, unchanged,
for the three primary arms, and estimates ten-seed D and the encoder contrast with seeds resampled. E4's
confirmatory results, with three seeds, are not changed and remain the ones reported as pre-registered; the
ten-seed values are reported beside them as exploratory.

**Known at the time:** E4's three-seed results on BGE (D, the encoder contrast, the neighbour-level ratios)
and the Qwen ten-seed results. No BGE run with any of the seven seeds existed.
