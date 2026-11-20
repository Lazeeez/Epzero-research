# Research status and decisions

## Scope of this implementation

The first delivery establishes executable correctness controls and a synthetic three-snapshot
pilot. It does not certify G0, G1, or G2 as fully passed under the supplied requirements. Research
conclusions, novelty claims, and full-data runs remain pending.

Since that first delivery, the native engine has been built from source in Ubuntu: 115 C++ tests
and 106 Python/research tests passed, and two fresh-process pilots matched. The real release audit
reproduced query/qrel populations and enumerated all three dense identifier maps. See
[NATIVE_VALIDATION.md](../data/NATIVE_VALIDATION.md) and [RELEASE_AUDIT.md](../data/RELEASE_AUDIT.md).

Follow-up on 2026-11-16: WSL now exposes 23.47 GiB RAM. Resumable full archive ingestion,
memory-mapped vector validation, ID-joined vector comparisons, a blocked depth-100 oracle, and
bounded insertion/state fingerprinting are implemented. The updated Linux research/IR suite has
85 passing tests; all 153 pilot run files and state fingerprints still match the earlier source
build. The full real-data preparation job is running; its vector-integrity result is pending.
See [INGESTION.md](../data/INGESTION.md) for live artifact locations and precise scope.

Further progress: snapshot 1's full archive and all 869,902 vectors are now verified; all vectors
are finite, nonzero and unit-normalized within `1e-5`. The full audit used about 274 MiB peak RSS.
Fresh-process real-vector resource probes completed at 8,000 and 64,000 indexed documents, and
the expanded Linux suite passes 94 tests. Text/model downloads and full longitudinal checks are
still in progress. See [RESOURCE_VALIDATION.md](../data/RESOURCE_VALIDATION.md). PACE job scripts are
prepared; cluster authentication and resource inventory remain pending, with no jobs submitted.

Development decisions made now:

| Question | Decision | Boundary |
|---|---|---|
| Initial representation | Synthetic float32 L2 vectors | Real encoder/release selection remains open |
| Construction | Manual uncompressed HNSW, one thread, fixed settings | No automatic build-plan changes |
| Restart strategy | Fresh deterministic replay | No native save/load continuation |
| Independent implementations | FAISS HNSW and IVF-Flat | Including IVF in this cheap pilot does not settle final paper scope |
| Query population | Explicit fixture evaluation IDs; real qrel cohorts enumerated | Official evaluator population semantics remain unresolved |
| Operating regimes | Frozen settings and matched calibration recall | Matched cost remains open |
| Insertion chronology | Lexical new IDs within each snapshot | Declared retrospective order, not production chronology |
| Practical effect margin | Not chosen | Must be fixed before final inferential analysis |

## Acceptance-test coverage

These rows describe implemented coverage, not blanket passes of the original T01–T20 contracts.
Use the commands in README.md and [VALIDATION.md](../data/VALIDATION.md) for actual execution evidence.

| Contract | Current evidence / remaining work |
|---|---|
| T01 snapshot integrity | Fixture mutation controls pass; real stored-ID sets are unique and cumulative, but rows reorder; real shared text/vector consistency remains open |
| T02 IDs/outputs | Stable string mapping; duplicate/padded/nonlive IDs and short/nonfinite outputs rejected; broader real-ID audit pending |
| T03 representation | Fixture vector checks; actual archive metadata identifies 2,560 dimensions and exact stored counts; real encoder/vector compatibility pending |
| T04 leakage | Cross-role fixture IDs/text/vectors rejected; actual 100/381 train/test split verified and provisional 50/50 training roles recorded; real vector leakage checks pending |
| T05 evaluator parity | Independent pytrec_eval parity passes; real qrel counts/cohorts reproduced with checksums; official population semantics pending |
| T06 continuation | Single-thread Epzero replay topology/rank equality; FAISS replay ranks checked; native checkpoint repair and compressed cases pending |
| T07 freeze | Explicit settings and validated configuration; full cross-machine/changed-k acceptance pending |
| T08 exact references | NumPy vs FAISS Flat and Epzero exact for L2/IP/cosine; blocked depth-100 oracle agrees on test inputs, including boundary ties; full real references and compressed/half extensions pending |
| T09 barriers/failures | Synchronous stages, fail-closed inputs, failed manifests, no-update repeats; mutation/crash injection incomplete |
| T10 instrumentation | Component timings and sampled process RSS only; lightweight native counters/overhead gate pending |
| T11 maintenance | Fresh IVF training vs frozen training IDs recorded; other intervention isolation pending |
| T12 precision | Sub-micro score separation, negative/scientific scores, ties, and round-trip relevance regression tested |
| T13 diagnostics | Strict/tie-aware ANN recall and judged fraction; symmetric-difference judgment coverage pending |
| T14 inference | Not implemented; no significance/equivalence claims |
| T15 trajectories | All nine arms through three synthetic snapshots; source-built three-seed pilots match across fresh processes; live sets, ancestry, replay, and training scopes checked |
| T16 resources | Development component ledger; isolated peaks, capacity-growth and budget-exhaustion gates pending |
| T17 calibration | Explicit k, exhaustive grids, nonmonotone curves, target misses and repeat selection; matched-cost gate pending |
| T18 provenance | Input/config/code/binary/output hashes; identity-bound resumable archive downloads, checksum rejection, staged extraction and failed job records; broader recovery/failure injection incomplete |
| T19 guard | Not in the core pilot; deferred |
| T20 reproduction | CLI fixture/run/verify and CI recipe; independent full-data reproduction pending |

## Next research milestones

1. Resolve historical model/embedding provenance and compatible query encoding. The metadata/ID
   audit does not establish unchanged vectors or text for shared documents. Finish G0 with this evidence.
2. Finish the running full-corpus ingestion and ID-joined vector audit. WSL has 23.47 GiB RAM, but
   the fixture runner still retains all snapshots/arms, and IVF training still copies all training
   vectors. Build an isolated full-data runner and an explicit feasible IVF training contract.
3. Execute real depth-100 references after query compatibility is established; add isolated resource
   measurement, matched-cost regimes, and broader interruption/failure tests before G2/G3 signoff.
4. Measure full-data feasibility, then lock the research matrix, practical effect margin, and paired
   query-trajectory analysis. Core experiments follow those gates.
5. Complete the close-prior-work review and an evidence-to-claim matrix before drafting novelty claims.

The supplied documents remain proposals and source material. Their twelve-week outline is not an
execution promise, and no current shared-task deadline or eligibility has been asserted here.
