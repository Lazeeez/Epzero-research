# Independent NGFix fitting/tuning screen — frozen before PACE execution

Status update: **job 7148412 returned and all three split results verify locally**;
see [NGFIX_DISJOINT_RESULTS.md](NGFIX_DISJOINT_RESULTS.md), including the cross-build
raw-state identity limitation. All 24 selected target points meet development mean
recall. The protocol below was frozen before execution and remains unchanged. This
corrects evaluation; it is not a novel algorithm or a fair-training-budget claim.

The verified original screen used 50 queries both to fit repair and select ef.
The new protocol keeps all 50 development queries fixed and divides the old
calibration pool into 30 fitting and 20 tuning queries. IDs are ordered by
SHA-256(`longeval-ngfix-disjoint-v1|` + query ID), with deterministic rotations
of 0,16 and 33 positions for splits 0,1 and 2. The first 30 fit; the remaining 20 tune.
Neither embeddings, exact neighbors nor performance determine assignment.
No query is shared between roles within a split. Splits overlap across runs.

For each of base, NGFix, RFix and both, select the smallest ef in 100/200/400/800
whose **tuning** mean recall reaches.95 or.99. If none qualifies, report no selected
setting. Report development recall and whether it reaches the target, even when
it misses. Do not select again on development or replace a failed target point.

The fitting-reference matrix sent to C++ has exactly 30 rows, ordered with the
first 30 query vectors. Tuning/development exact references are available only to
the external verifier. The existing C++ implementation and original 500 package
files are unchanged. `ngfix_disjoint.py` adds semantic roles and selection over the
unchanged score/ranking verifier. Its tests prove that perfect fitting or development
performance cannot force ef 100 when tuning requires ef 200, and that an unattainable
tuning target remains unattained.

The old mechanical verifier still writes legacy fit-based selections into
`verification.json`; these are retained for reproducibility and must not be used
as the new result. **`disjoint-verification.json` is authoritative for tuning
selection.** Local return verification recomputes its groups and choices from raw
rankings/timings and checks the frozen parent-derived role contract.

Thirty fitting queries versus the original 50 is a deliberate training-budget
change. Therefore a difference from job 7141656 cannot be attributed solely to
selection independence. These three overlapping splits are exploratory sensitivity
checks, not three independent datasets or construction seeds. Twenty tuning queries
are a small population, and broader independent training remains a paper requirement.

PACE executes 19 relevant Python tests, three native rejection checks, then the
new disjoint-role synthetic fixture before downloading the corpus. All Slurm steps
start explicitly in `/tmp` and use absolute paths; execution reports capture actual
CWD, affinity and mount view. Every large download, matrix and graph stays on
compute-node temporary disk. Graph bytes are not returned, only their checked
hashes/sizes. Run at most this one prepared job; no full-data sweep is justified yet.

Package: `build/longeval-ngfix-disjoint-01.tar.gz` (2,556,701 bytes, 506 files), SHA-256:
`8f2593bdacd467a448ff08dfd64e1bb8bc97f491e2dfc750f4ef979618d023f0`.
Frozen protocol SHA-256:
`b61406b9758eeaf539a61c41baabf34e817c89cc5c3490ee1612885b837bbb2e`.

Local evidence: `build/longeval-ngfix-disjoint-inputs-01/preflight.json`,
`build/ngfix-disjoint-linux-tests-01.xml`, and the package receipt. A launch from
the old checkout succeeds with Python `-P`; Linux preparation reproduces the
Windows-prepared split 0 input/protocol hashes byte for byte. Source, data and
earlier result archives remain preserved. The earlier 57-file NGFix result state
was archived before these status edits as
`build/longeval-ngfix-results-7141656-preserved.tar.gz`, SHA-256
`4bbe0dda593a3df69ceb609212bbbc2c9a3627f0b2475814d4dc7ccecc241a9e`.

See [submission and retrieval commands](../../../../scripts/pace/LONGEVAL_NGFIX_DISJOINT.md).
The 381 final held-out query retrieval outcomes remain untouched.
