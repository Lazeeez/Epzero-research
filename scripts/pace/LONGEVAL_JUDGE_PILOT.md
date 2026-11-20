# Judge pilot job (GPU)

An exploratory follow-up to the held-out study, not part of `FINAL_PROTOCOL_V1`. One GPU job runs
GPT-OSS-120B locally as a relevance judge on 270 prompt-development pairs, to find out whether the
judge runs, whether its answers parse, how fast it is and how stable its grades are.

It decides nothing about the study. The pairs are officially graded pairs from outside the
study's own pool, and they are excluded from the later validation sample. The validation
thresholds are in `research/longeval/docs/protocol/JUDGE_VALIDATION_THRESHOLDS.md`; this job does
not test them.

## What the judge does

The judge follows the procedure that produced the lab's released LLM labels
(`clef26/topic-extraction/evaluation` in <https://github.com/clef-longeval/longeval-code>,
commit `b5f11fd2d58e`):

| Step | Organisers | This job |
| --- | --- | --- |
| Input | Query text; document title, a space, the abstract | The same, from the released files |
| Prompt | `umbrella_zeroshot_no_desc_no_narrative` | The same text, character for character (a test pins its SHA-256) |
| Call | One user message, no system message, no sampling parameter | The same request, plus a 4,096-token cap on the answer |
| Grade | UMBRELA's answer parser on the message content; 0 if nothing matches | The same 24 patterns in the same order (pinned by a test) |
| Model | `openai/gpt-oss-120b` through a hosted endpoint | The published weights at one pinned revision, served locally |

Two things cannot be matched and are what the pilot measures:

- **The serving stack.** The organisers called a hosted endpoint whose settings are not recorded.
  This job serves the published MXFP4 weights with vLLM 0.33.0 on one GPU.
- **Sampling.** The organisers sent no temperature, so the provider's default applied and their
  grades are single samples. The job grades every pair four times: twice with nothing sent
  (`provider_default`) and twice at temperature 0 (`temperature_0`). The repeats show how much a
  grade changes between runs, which is needed before the sampling setting is pinned for validation.

The organisers' parser takes the first of its patterns that matches anywhere in the answer, so a
phrase such as "from 0 to 2" can override the category on the last line. The parser is kept as
they ran it, and the report counts how often it differs from the last line.

## Before submitting

The package needs an NVIDIA H200 (the job checks for at least 100,000 MiB), a driver of version
580 or newer for the pinned CUDA 13 packages, and 110 GiB free on the node's temporary disk. A
one-minute check on an H200 node:

```bash
srun --account=coc --qos=coc-ice --partition=coc-gpu,ice-gpu --gres=gpu:h200:1 --tmp=160G --time=00:05:00 \
  bash -c 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h "${TMPDIR:-/tmp}" | tail -n 1'
```

## Submit

Copy `longeval-judge-pilot-01.tar.gz` to the PACE LongEval checkout, compare its SHA-256 with
`longeval-judge-pilot-01-receipt.json`, then:

```bash
sha256sum longeval-judge-pilot-01.tar.gz
tar -xzf longeval-judge-pilot-01.tar.gz
(cd longeval-judge-pilot-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
sbatch longeval-judge-pilot-01/scripts/pace/longeval-judge-pilot.sbatch
```

Resources: one H200, 8 CPUs, 96 GB RAM, 160 GB node-local temporary disk, 4-hour ceiling. Expect
about one hour, most of it setup: the Python packages (199 pinned wheels), the 65 GB model
download and the model load. The judging itself is 1,080 requests, 16 at a time. Everything large
stays on the node's temporary disk.

## What the job does, phase by phase

The log prints `phase=...` as it goes and `exit_code=... phase=...` at the end, so a failure names
its phase.

| Phase | What happens | If it fails |
| --- | --- | --- |
| `package` | Verifies the package checksums | The archive was changed or truncated in transit |
| `gpu` | Checks GPU memory and driver version | Another GPU type was allocated, or the driver is older than 580 |
| `environment` | Sets up the node-local staging directory and checks free space | The temporary disk is shared storage or too small |
| `tests` | Runs the judge's tests and validates the inputs | The package does not match its inputs |
| `gpu-environment` | Installs the pinned serving stack into a fresh environment | A pinned wheel is unavailable for this platform |
| `model` | Downloads the pinned model revision and verifies every file by SHA-256 | Network trouble (the same job can be resubmitted) or a changed file |
| `server` | Starts the model server and waits until it answers | See `server-log-tail.txt` in the evidence |
| `probe` | Sends one judgment and checks that the answer arrives separately from the reasoning | The server returns the answer in an unexpected shape; see `judging/probe.json` |
| `judge` | Grades the 270 pairs under both settings, twice each | Requests that fail three times are recorded, not fatal |
| `analyze` | Writes `judging/report.json` | — |

## After completion

```bash
job=JOB_ID
sacct -X -j "$job" --format=JobID,JobName%32,State,ExitCode,Elapsed
tail -n 25 "longeval-judge-${job}.log"
tar -czf "longeval-judge-${job}-evidence.tar.gz" \
  -C build/pace-longeval-judge-pilot/jobs "$job" -C "$PWD" "longeval-judge-${job}.log"
sha256sum "longeval-judge-${job}-evidence.tar.gz"
```

The archive is a few megabytes. Copy it to the local `build` directory. It holds the model's
answers for LongEval query–document pairs and must never be committed to the public repository.

## What comes back

`judging/report.json` gives, for each setting and repeat:

- how many pairs were answered, how many answers had no final content and how many did not parse;
- how often the organisers' parser differs from the category on the last line;
- the grade distribution, and the agreement with the official grades of these pairs (exact,
  binary and linear-weighted kappa, with query-clustered intervals);
- the agreement between the two repeats of a setting, and between the two settings;
- token counts and pairs per hour, which size the validation and pool-grading jobs.

The pilot pairs are stratified equally by official grade, so its agreement figures describe the
pilot sample and not the label population. They are for choosing a configuration, not for the
threshold decision.

## What the pilot decides

Only the configuration to pin before validation: the sampling setting, the answer cap and the
serving stack. Once that is pinned and the advisor has approved the thresholds, the validation
sample (1,000 pairs, excluding these 270) is drawn and graded in a separate job.
