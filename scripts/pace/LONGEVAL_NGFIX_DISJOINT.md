# NGFix independent fitting/tuning screen

One CPU job, three overlapping deterministic splits of the already-exposed sample.
Each split uses30 fitting,20 tuning and50 unchanged development queries. Only tuning
selects ef; development misses remain reported. This is an exploratory correction,
not a new-method benchmark or a properly scaled training-budget comparison.

The frozen `protocol.json` contains all roles. The published native algorithm,
compiler flags, sample and100 query vectors remain unchanged. Original screen03
files are carried unchanged; the new module interprets legacy native input names
explicitly. The old `verification.json` contains fit-based legacy selections:
**use `disjoint-verification.json` for the independent tuning selections.**

Copy `build/longeval-ngfix-disjoint-01.tar.gz` to the existing isolated PACE Epzero
checkout, compare its SHA-256 with `build/longeval-ngfix-disjoint-01-receipt.json`,
and run from that checkout:

```bash
sha256sum longeval-ngfix-disjoint-01.tar.gz
tar -xzf longeval-ngfix-disjoint-01.tar.gz
PYTHONPATH="$PWD/longeval-ngfix-disjoint-01" build/pace-longeval/venv/bin/python \
  -P -m research.longeval.ngfix_screen package-verify "$PWD/longeval-ngfix-disjoint-01"
sbatch longeval-ngfix-disjoint-01/scripts/pace/longeval-ngfix-disjoint.sbatch
```

This requests one exclusive Gold6226 node, one compute thread,8GB RAM,48GB node-local
temporary storage and2h ceiling. It runs19 Python tests, three native rejection
checks and a disjoint-role synthetic native fixture before downloading anything
large. The single8.89GB corpus archive, extracted vectors, sample matrices and
all indexes stay in the compute node's temporary directory. No shared/scratch
download fallback exists. Each step starts explicitly in `/tmp`, uses absolute
paths, and records the actual execution directory, CPU affinity and mount view.

All three real runs have1,600 conditions and4,800 timed calls. Changing which30
queries fit the repair is intentional. Do not treat these overlapping splits as
independent datasets, or attribute differences from the original50-query fit
entirely to tuning separation. The exact native fitting-reference matrix contains
only fitting queries; tuning/development references stay in the external verifier.

After completion:

```bash
job=JOB_ID
sacct -X -j "$job" --format=JobID,JobName%32,State,ExitCode,Elapsed
tail -n 50 "longeval-ngsplit-${job}.log"
tar -czf "longeval-ngsplit-${job}-evidence.tar.gz" \
  -C build/pace-longeval-ngfix-disjoint/jobs "$job" \
  -C "$PWD" "longeval-ngsplit-${job}.log"
```

Copy that evidence archive into the local `build` directory. Keep failure evidence.
Expected results are still unknown; `COMPLETED` is not itself scientific success.
The381 final held-out retrieval outcomes are not accessed.
