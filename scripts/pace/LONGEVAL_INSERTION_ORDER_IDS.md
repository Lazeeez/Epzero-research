# Insertion order at full scale, again: returned IDs and confirmation queries

An exploratory follow-up to `LONGEVAL_INSERTION_ORDER.md`, not part of `FINAL_PROTOCOL_V1`.
Recorded on 23 November 2026, before any of these jobs was submitted. Decided by the author; the
advisor was unavailable.

## Why

The first full-scale run (jobs 7208888–7208893) saved, per query, only how many exact neighbours each
build missed in each arrival group. That supports ratios and rate differences, but not three things
the paper needs:

1. **Paired, target-level analysis.** Following each (query, exact target) pair across the builds
   (found in both, missed in both, rescued, newly lost) needs the returned IDs.
2. **The relevance consequence of the reversal.** Grading what each order drops and puts in its
   place needs the returned documents.
3. **Confirmation on queries never used for any arrival-order result.** The full-scale result was
   seen on the 381 test queries, and the within-query analysis was chosen afterwards.

## Design

Identical to the first full-scale run for the study's graph (all of snapshot 3, 1,661,900 documents;
M 16, efConstruction 100, inner product, one thread; orders as arrived, reversed blocks and
shuffled; seeds 11, 29 and 47; both implementations; search widths 100, 200, 400, 800; exact
top-100 targets), with three additions:

- every build saves the returned top-100 sample positions at every width (`labels-ef*.npy`), and the
  job keeps the document IDs in sample order (`doc-ids.txt.gz`);
- the **100 training queries** are searched too, as the confirmation set. They share no ID or text
  with the test queries. 50 of them calibrated search widths in the held-out study; none has been
  used for any arrival-order outcome. Their snapshot-3 exact top-100 is computed in the same pass and
  checked against the study's development references (`confirm_reference_check`); a mismatch stops
  the job;
- the paired analysis (`order_pairs.py`) is tested on the node.

Six jobs (two implementations × three seeds), each building the three orders, one exclusive Gold 6226
node each, as before.

## Checks fixed now

- **Reproducibility.** With the same seed, order and code, the builds should reproduce the first
  run's per-query miss tables exactly (single-threaded, seeded construction). Any difference is
  reported and explained before analysis.
- **IDs agree with counts.** The saved IDs must reproduce the saved miss tables exactly (tested).

## Analyses fixed now

**A. Confirmation (primary for this follow-up).** On the 100 confirmation queries, ef 200, newest
arrival group (snapshot 3) against oldest (snapshot 1), per implementation, seeds averaged within
query, paired query bootstrap (`insertion_strata analyze --confirm`):

- **Primary:** the within-query rate difference, as arrived minus reversed. **Prediction:** positive
  for both implementations, with the 95% interval excluding 0. (On the test queries: +4.28 pp
  Epzero, +3.41 pp FAISS; with 100 queries the interval is about twice as wide.)
- **Secondary:** the within-query ratios (prediction: above 1 as arrived, within [0.85, 1.15]
  shuffled, below 1 reversed); overall recall@100 differing by less than 1 point between orders; the
  same at ef 100, 400 and 800.

A result against a prediction is reported as such.

**B. Paired target-level analysis** (`order_pairs.py`), on the test and on the confirmation queries:
for as arrived against reversed, as arrived against shuffled and shuffled against reversed, by
arrival group: pairs found in both, missed in both, rescued, newly lost; the paired change in miss
risk per group with its interval and per seed; and the change in the gap between the newest and the
oldest group.

**C. Relevance consequence** (after the E1 judging pool, extended with these runs' top-10
substitutions, is graded), test queries, ef 200, top 10, per order:

- the share of judged-relevant exact top-10 pairs dropped, by arrival group;
- the grades of the documents put in their place, and judged@10 by group;
- net ranked gain: nDCG@10 of the build minus nDCG@10 of exact search per query, under the official
  GPT-OSS-120B grades and under the local full grades, and its change between orders.

**D. Human audit (replaces the 21 November E3 sample for the paper).** Drawn once, after these runs
return and before any grade of the drawn pairs is compared with a role: from the as-arrived and
reversed builds of seed 11 at ef 200, 50 documents dropped from an exact top 10 that belong to
snapshot 3 and 50 that belong to snapshot 1 (whether or not the release graded them), and the 100
documents put in their place, sampled with seed 20261123 by a sampler added to `audit.py` before the
draw; presented blind to role, arrival group and order, on the 0–2 scale of the organisers' prompt.
A second annotator, if one is found before unblinding, grades a 50-pair overlap.

## Submit

Build the package locally:

```bash
python scripts/pace/package-longeval-insertion-order-ids.py BUILD/longeval-heldout-inputs-01 HELDOUT_JOBS BUILD
```

Copy `longeval-insertion-order-03.tar.gz` to the PACE LongEval checkout, compare its SHA-256 with the
receipt, then:

```bash
sha256sum longeval-insertion-order-03.tar.gz
tar -xzf longeval-insertion-order-03.tar.gz
(cd longeval-insertion-order-03 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
for backend in epzero faiss_hnsw; do for seed in 11 29 47; do
  sbatch longeval-insertion-order-03/scripts/pace/longeval-insertion-order-ids.sbatch "$backend" "$seed"
done; done
```

Keep all jobs on the same CPU type, and do not overlap them with jobs on other CPU types.

## After completion

```bash
sacct -X --name=longeval-order-ids -S 2026-11-23 --format=JobID,State,ExitCode,Elapsed
grep -H '^PASS\|^exit_code' longeval-order-ids-*.log
tar -czf longeval-insertion-order-03-evidence.tar.gz build/pace-longeval-insertion-order-ids/jobs longeval-order-ids-*.log
sha256sum longeval-insertion-order-03-evidence.tar.gz
```

The archive holds run files and document IDs derived from LongEval data: copy it to the local
`build` directory and never commit it.
