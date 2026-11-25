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
