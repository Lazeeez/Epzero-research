# Full-scale insertion-order jobs

An exploratory follow-up to the held-out study, not part of `FINAL_PROTOCOL_V1`. It tests at full
scale what the sample pilot found (`research/longeval/HELDOUT_EXPLORATORY.md`, section 10): an HNSW
index built by appending under-retrieves the documents that were inserted last, and reversing the
insertion order moves the excess onto the documents that were inserted first.

No relevance judgment is used or shipped. The outcome is neighbour recall against exact search, by
arrival group and by insertion decile.

## Design

One job builds one index three times, on the same documents with the same construction seed, in
three insertion orders:

| Order | Insertion sequence |
| --- | --- |
| As arrived | Snapshot-1 documents, then the additions of snapshot 2, then those of snapshot 3; lexical ID order inside each block. This is the order of the held-out incremental arm. |
| Reversed | The three blocks in the opposite order: snapshot 3, 2, 1. Lexical ID order is kept inside each block. |
| Shuffled | One random permutation of all documents, seeded by the build seed, so both implementations insert the same sequence for a given seed. |

Each build uses the study's graph settings (M 16, efConstruction 100, inner product, one thread)
and is searched with the 381 test queries at ef 100, 200, 400 and 800. Exact neighbours are
computed inside the dataset in float64.

| Dataset | Documents | Purpose |
| --- | --- | --- |
| `full` | All 1,661,900 documents of snapshot 3 | The confirming experiment |
| `draw1`, `draw2`, `draw3` | 869,902 documents each (the size of snapshot 1), keeping the arrival groups in proportion: 455,340 / 237,022 / 177,540 | How the result varies with collection size and with the documents drawn |

With two implementations and three seeds this is 6 jobs for `full` (18 builds) and 18 jobs for the
draws (54 builds).

## What will be estimated

These definitions are fixed here, before any full-scale build exists.

- **Miss ratio.** Misses of a later arrival group, divided by the misses expected at the
  snapshot-1 group's miss rate in each exact-rank band (1–10, 11–50, 51–100). It is reported for
  snapshot-3 arrivals, for snapshot-2 arrivals, and for both together.
- **Primary order effect.** The as-arrived ratio minus the reversed ratio at ef 200, on the same
  queries, documents and seed, with a paired query bootstrap (5,000 draws); per seed and as the
  mean over the three seeds.
- **Reversal.** A reversed ratio whose whole interval lies below 1. The pilot shows this for
  snapshot-3 arrivals against snapshot-1 arrivals and not for snapshots 2 and 3 together, because
  the snapshot-2 block sits in the middle of both orders. Which of the two counts as the reversal
  test is an open decision for the advisor; both are reported.
- **Position alone.** In the shuffled build, the miss ratio of each insertion decile against the
  first.
- **Connectivity.** Base-layer incoming links by insertion decile and by arrival group.

The pilot results are known and are prior evidence. Nothing here changes the frozen analysis.

## A recorded cross-check

For `full`, the as-arrived build inserts the same documents in the same order with the same seed
as the held-out incremental arm, but through different code. The job compares its per-query
neighbour recall at ef 200 with the held-out run and writes the result to
`heldout-comparison.json`. It is recorded, not required: identical graphs are expected only where
both code paths insert one document at a time in the same way.

The sample preparation does have a hard check: for `full`, the exact neighbours it computes must
agree with the held-out exact references, or the job stops before building anything.

## Submit

Copy `longeval-insertion-order-01.tar.gz` to the PACE LongEval checkout, compare its SHA-256 with
`longeval-insertion-order-01-receipt.json`, then:

```bash
sha256sum longeval-insertion-order-01.tar.gz
tar -xzf longeval-insertion-order-01.tar.gz
(cd longeval-insertion-order-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)

# The confirming experiment: six jobs.
for backend in epzero faiss_hnsw; do for seed in 11 29 47; do
  sbatch longeval-insertion-order-01/scripts/pace/longeval-insertion-order.sbatch full "$backend" "$seed"
done; done

# The draws: eighteen jobs. Submit them when the six above have passed.
for dataset in draw1 draw2 draw3; do for backend in epzero faiss_hnsw; do for seed in 11 29 47; do
  sbatch longeval-insertion-order-01/scripts/pace/longeval-insertion-order.sbatch "$dataset" "$backend" "$seed"
done; done; done
```

Resources per job: one exclusive Gold 6226 node, one thread, 64 GB RAM, 160 GB node-local
temporary disk, 12-hour ceiling. Expect four to five hours for `full` (about an hour of download,
extraction and audit, a few minutes for the sample, and roughly an hour per build) and two to
three hours for a draw. As with every CPU job of the study, keep all jobs on the same CPU type.

## After completion

```bash
sacct -X --name=longeval-order --format=JobID,JobName%20,State,ExitCode,Elapsed
grep -h '^PASS\|^exit_code' longeval-order-*.log
tar -czf longeval-insertion-order-01-evidence.tar.gz build/pace-longeval-insertion-order/jobs longeval-order-*.log
sha256sum longeval-insertion-order-01-evidence.tar.gz
```

The archive is small: reports and per-query count tables, no vectors and no document
identifiers. Copy it to the local `build` directory.

## Local analysis

```bash
# Gather the builds of one dataset, verifying each file against its job's checksum list.
python -m research.longeval.insertion_pilot collect EVIDENCE/build/pace-longeval-insertion-order/jobs full build/insertion-order-full
# Ratios, paired contrasts and decile curves at every search width.
python -m research.longeval.insertion_pilot robustness build/insertion-order-full/sample \
    build/insertion-order-full.json build/insertion-order-full
```

`collect` refuses a job that did not end with exit code 0, a file that does not match its
checksum, and two jobs that return the same build.
