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
