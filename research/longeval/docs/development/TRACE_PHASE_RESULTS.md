# Full-data search-phase audit — 2026-11-18

All **1,800 complete traces / 30,126,016 scored-edge events** were re-read and
checked against their original event hashes, PACE artifact receipts, exact-neighbor
references and distance-call counts. This covers three snapshots, seeds 11/29/47,
100 exposed training queries and native ef=200/3200. The scanner has seven passing
tests. No held-out query retrieval was performed.

Evidence: `build/longeval-trace-phases-01/report.json`, `per-query.json`, and
`exact-id-map.json`; implementation: `trace_phases.py`; execution script:
`build/analyze-longeval-trace-phases.py`.

## Finding: most remaining errors occur after a true neighbor is found

Final-snapshot development results (50 queries, exact k=100):

| Seed | ef=200 missing neighbor occurrences | Missing because no exact neighbor encountered | Missing at wide-search 4,000-call prefix | Zero-encounter queries at that prefix |
| --- | ---: | ---: | ---: | ---: |
| 11 | 400 | 200 | 191 | 0 |
| 29 | 209 | 0 | 166 | 0 |
| 47 | 206 | 0 | 166 | 0 |

All **523** final-development missing occurrences at the previously selected
4,000-call prefix are unencountered neighbors of queries that have already found
some exact neighbors. There are no encountered-but-omitted exact neighbors in
these final-development conditions. This supports investigating neighborhood
coverage, beyond the two already diagnosed entry failures. It does not establish
that changing the entry could never help later search.

At native ef=200, the median first layer-0 exact-neighbor discovery takes
146.5/143/145 total distance calls; the seed-11 median excludes its two failures,
which are retained explicitly. On the wide trace, all queries reach 95 exact
neighbors, with median calls 1,582.5/1,493.5/1,537.5, but maxima
27,556/36,675/35,974. The same four development queries exceed 10,000 calls in all
three seeds. These are correlated observations on the same exposed query set,
not independent confirmation. Do not select method parameters using those IDs.

Across *all* snapshots/roles/efforts, 18 exact-neighbor occurrences were encountered
but absent from the final returned top-100, and six such occurrences appear at the
4,000-call prefix. The report preserves these discrepancies; encounter counts are
candidate-coverage bounds, not automatically identical to achieved recall.
Their numerical/tie causes have not been separately established by this audit.

## Document age is not a clean repair selector

Final development exact answers contain 3,232 snapshot-1, 1,058 snapshot-2 and 710
snapshot-3 occurrences. At ef=200, cohort miss rates are:

| Seed | Snapshot-1 documents | Added in snapshot 2 | Added in snapshot 3 |
| --- | ---: | ---: | ---: |
| 11 | 7.86% | 7.09% | 10.00% |
| 29 | 3.47% | 4.63% | 6.76% |
| 47 | 3.56% | 4.35% | 6.34% |

Newer answers have higher miss rates at this operating point, but the trend
reverses with wide search: snapshot-1 miss rates are 0.526–0.588%, while
snapshot-3 rates are 0–0.141%. Most absolute wide-search misses are older documents.
Birth cohort is confounded with corpus/query geometry. This does not justify
an age-only repair policy or revive an index-age degradation claim.

## Scope and implications

Labels were joined through the lexically sorted final-snapshot external-ID map,
SHA-256 `78509abe8277f17fb964e3ca099552bbc80745f1d16aed1b5780cb7523bca5f9`.
All three local document maps were checked against ingestion hashes. Snapshot row
positions were not used as stable document identities.

Milestones are retrospective oracle measurements. The logs contain scored edges,
not every adjacency or expansion; they cannot prove missing links, disconnected
components, or the outcome of a different search policy. No native speedup follows
from distance-call counts. The two-phase interpretation also already exists in
[NGFix/RFix](https://arxiv.org/html/2510.22316v1); this audit is diagnosis, not novelty.

The next method must improve neighborhood coverage with a distinct mechanism and
transfer beyond fitted answer IDs. Compare directly with NGFix/RFix, wider search
and strong standard graphs; include extra-edge memory and fitting/maintenance work.
Do not promote another historical-query shortcut variant without that distinction.
