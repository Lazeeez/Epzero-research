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
