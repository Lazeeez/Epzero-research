# Design record: insertion order at full scale

**Written on 21 November 2026, after the jobs were submitted and before any of their output was
returned or looked at.** It is a record, not a freeze: it states what was fixed, where, and when,
so that the full-scale result can be read as a test of a design that existed first.

## What was fixed before the jobs ran

The design, the datasets and every estimate are defined in
`scripts/pace/LONGEVAL_INSERTION_ORDER.md`, section "What will be estimated". The files below
were committed and pushed before the jobs were submitted, and the package the jobs run was built
from them.

| File | Last commit | Committed | SHA-256 |
| --- | --- | --- | --- |
| `scripts/pace/LONGEVAL_INSERTION_ORDER.md` | `51e9bfe9234d` | 2026-11-21T11:03:36+09:00 | `277ef7c3e8d0a7c39425379e45ef2e2a6e2117d884db331270658b1d2a64baf3` |
| `scripts/pace/longeval-insertion-order.sbatch` | `86f5c6dc9cb6` | 2026-11-21T11:03:36+09:00 | `715df24c79ab63f3864da144e96f986a25ec140d624158bd95b50e9fc38edb74` |
| `scripts/pace/package-longeval-insertion-order.py` | `b8c9328fb39f` | 2026-11-21T11:03:36+09:00 | `e79bd2994e556364027f917b45b8a569349426fe146218b23239689075217bc7` |
| `research/longeval/insertion_pilot.py` | `c078ddf37a8e` | 2026-11-21T11:03:36+09:00 | `62c286ba2c70924a234ba691fe020e48f927993b45e47cbc6b46efa2d9ed3f9b` |

- Package: `longeval-insertion-order-01.tar.gz`, SHA-256 `fa99334e448311a71de201c606c1cdc2626d9a5b5adba51b82d9026db91a1718`.
- Jobs: 7208888–7208893 (all of snapshot 3; two implementations and three seeds), submitted on
  21 November 2026. They were queued when last checked, from the scheduler's own listing.
- The draw jobs (`draw1`, `draw2`, `draw3`) had not been submitted.

## The contrasts

- **Primary.** The as-arrived miss ratio minus the reversed miss ratio at ef 200, for snapshot-3
  arrivals against snapshot-1 arrivals, per implementation, as the mean over three seeds with a
  paired query bootstrap (5,000 draws).
- **Reported beside it.** The same contrast for snapshots 2 and 3 pooled; the three ratios
  themselves; ef 100, 400 and 800; misses and neighbours by rank band, arrival group and order,
  with absolute miss rates; the decile curve of the shuffled build; incoming links by decile.
- **Open.** Which of the two contrasts counts as the reversal test is an advisor decision
  (`EXTENSION_PROTOCOL_DRAFT.md`). Both are reported whatever is decided.

## What was already known

The pilot on a 300,000-document sample (three orders, four search widths, two graph settings,
insertion deciles) and its replication on GloVe and SIFT, all in `HELDOUT_EXPLORATORY.md`,
section 10. The full-scale result is a confirmation of that pilot, on the same test queries, and
is reported as a prespecified follow-up. A full-scale result that disagrees with the pilot is
reported as such.
