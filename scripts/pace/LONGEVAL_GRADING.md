# E1: grading the judging pool with the local judge

Recorded on 25 November 2026, before the job was submitted and before any study pair had a local grade.
The design is section E1 of `research/longeval/docs/protocol/EXTENSION_PROTOCOL_V1.md` (decisions 1-3) and
amendment 5 of `EXTENSION_AMENDMENTS.md`.

## What the job does

On one NVIDIA H200 node, with the judge pilot's serving stack, kernels, model revision and server
settings (`LONGEVAL_JUDGE_PILOT.md`; job 7219729 passed with them): GPT-OSS-120B with the organisers' prompt
and parser (`judge.py`), temperature 0 (decision 1), an answer cap of 4,096 tokens, and every unparsed answer
asked again at most twice. Each grading folder records its configuration before the first request and
refuses to resume under another (`coverage.grade`).

1. **Substitutions** (`inputs/grading/substitutions`, 22 pairs): the order rerun's top-10 substitutions
   outside the primary pool (amendment 5). Graded first, as a live check: if any of them has no usable
   answer, the job stops before the primary pool.
2. **Primary** (`inputs/grading/primary`, 7,135 pairs): the 6,298-pair pool of decision 3 and the 1,000-pair
   validation sample (163 pairs in both), prepared by the frozen `coverage.prepare`.

From the pilot's throughput (5,000-5,600 pairs an hour at 16 parallel requests), grading takes about 1.5
hours after the environment and the 65 GB model are ready.

## After completion

```bash
sacct -X --name=longeval-grading -S 2026-11-25 --format=JobID,State,ExitCode,Elapsed
grep -H '^PASS\|^exit_code' longeval-grading-*.log
tar -czf longeval-grading-01-evidence.tar.gz build/pace-longeval-grading/jobs longeval-grading-*.log
sha256sum longeval-grading-01-evidence.tar.gz
```

Locally, `python -m research.longeval.coverage analyze` on the primary folder gives the validation against
the gate of decision 2 (binary agreement at least 0.90, linear-weighted kappa at least 0.80, nDCG@10 difference
at most 0.02 at every snapshot) and the coverage estimates; the substitutions' grades go to the arrival-order
analyses. The archive holds LongEval texts and grades: never commit it.

## Submit

Build the package locally (`python scripts/pace/package-longeval-grading.py BUILD_DIR`), copy
`longeval-grading-01.tar.gz` to the PACE LongEval checkout, compare its SHA-256 with the receipt, then:

```bash
tar -xzf longeval-grading-01.tar.gz
(cd longeval-grading-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
sbatch longeval-grading-01/scripts/pace/longeval-grading.sbatch
```
