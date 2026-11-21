# Judge pilot: results

**Job 7219729, 21 November 2026.** This is the pilot described in
[`scripts/pace/LONGEVAL_JUDGE_PILOT.md`](../../../../scripts/pace/LONGEVAL_JUDGE_PILOT.md). It ran
the local judge on 270 prompt-development pairs. It is not the validation of
[`JUDGE_VALIDATION_THRESHOLDS.md`](JUDGE_VALIDATION_THRESHOLDS.md): the pairs are not the
validation sample, they are balanced by grade, and no threshold is decided here. Its purpose is
to show that the judge runs, how repeatable it is, and which sampling setting to pin.

This file holds aggregate numbers only. The model's answers stay in the evidence archive and are
never committed.

## What ran

| Item | Value |
| --- | --- |
| Package | `longeval-judge-pilot-02.tar.gz`, SHA-256 `246d8d2fbac32176b6237a3e8040a5861daa4d9b8418b5bb108e90653a8bb2a6` |
| Evidence | `longeval-judge-7219729-evidence.tar.gz`, SHA-256 `4013158486b7a7913d242de757f73d3b3c3acc3c5fd2114b3d6f84dfa96cfcfa`, identical on PACE and locally |
| Integrity | Exit code 0, final phase `done`; the judging files match their checksum list; the job ran the manifest of package 02 |
| Model | `openai/gpt-oss-120b`, the pinned revision, every file verified by SHA-256 |
| Hardware | One NVIDIA H200 (143,771 MiB), driver 615.71.09; 130,209 MiB in use after loading |
| Serving stack | vLLM 0.33.0 with the pinned packages and the prebuilt kernels of `judge-kernels.txt`; no kernel was compiled at startup |
| Prompt and parser | The organisers' prompt (SHA-256 `7407ffe0…839b34`) and answer parser, one user message, no system message |
| Pairs | 270 prompt-development pairs over 178 queries: 30 per official grade (0, 1, 2) and snapshot |
| Runs | Two sampling settings, each run twice: no sampling parameter (the server's default), and temperature 0 |
| Elapsed | 23 min 46 s in all; about 3 minutes per run of 270 pairs at 16 parallel requests |

The first attempt (job 7205655, package 01) stopped before any request, as recorded in the
runbook. No judgment exists from it.

## Every answer was usable

All 1,080 requests were answered at the first attempt and ended normally. Every answer parsed to
a grade, and every answer ended with a single category on its last line. The organisers' parser
and the last line gave a different grade in 0 to 2 answers per run. The longest answer used 773
tokens against a cap of 4,096 (mean 241), so the cap was never reached.

## How repeatable the judge is

Agreement between the two runs of the same setting, on the same 270 pairs, with 95% bootstrap
intervals:

| Setting | Same grade | Same side of relevant / not relevant | Linear-weighted kappa | Identical answer text |
| --- | --- | --- | --- | --- |
| No sampling parameter | 0.874 [0.835, 0.912] | 0.930 [0.899, 0.959] | 0.856 [0.806, 0.899] | 1.1% |
| Temperature 0 | 0.915 [0.881, 0.948] | 0.952 [0.924, 0.977] | 0.907 [0.869, 0.943] | 11.1% |

- **Neither setting is deterministic.** At temperature 0 the grade still changes between two runs
  for 8.5% of pairs. A grade from this judge is a sample, whichever setting is used.
- **Temperature 0 is more repeatable, but the pilot does not settle by how much.** Paired on the
  same pairs, its share of identical grades is higher by 0.041 [−0.004, +0.085].
- **Disagreements are small.** Two runs that differ are one grade apart in 56 of 57 cases.
- All four runs give the same grade for 215 of the 270 pairs.

## Agreement with the organisers' grades

Each run against the released GPT-OSS-120B grade of the same pair:

| Setting, run | Same grade | Binary agreement | Linear-weighted kappa | Local grades 0 / 1 / 2 |
| --- | --- | --- | --- | --- |
| No sampling parameter, 1 | 0.896 | 0.959 [0.934, 0.981] | 0.884 [0.839, 0.925] | 89 / 88 / 93 |
| No sampling parameter, 2 | 0.867 | 0.919 [0.884, 0.949] | 0.847 [0.792, 0.893] | 92 / 85 / 93 |
| Temperature 0, 1 | 0.870 | 0.930 [0.899, 0.959] | 0.852 [0.804, 0.897] | 95 / 78 / 97 |
| Temperature 0, 2 | 0.870 | 0.926 [0.895, 0.955] | 0.856 [0.810, 0.899] | 98 / 79 / 93 |

The official grades are 90 / 90 / 90 by construction.

- **The local judge agrees with the organisers' grades about as well as it agrees with itself.**
  Agreement with the official grade (0.87–0.90) is in the range of the agreement between two
  local runs (0.87–0.92). This is what a faithful reproduction of a sampled assessor looks like:
  the released grade is one more sample of the same process.
- **The two settings do not differ in agreement with the official grades.** The differences
  between runs of one setting are as large as the differences between settings.
- **Temperature 0 gives fewer middle grades** (78–79 against 85–88), and its agreement on pairs
  officially graded 1 is 0.74 against 0.78–0.83.

**A forecast, not a result.** Reweighting each run's confusion matrix to the grade shares of the
released judgments (34% / 20% / 46%) gives a binary agreement of 0.935–0.966 and a kappa of
0.877–0.903. The validation thresholds are 0.90 and 0.80. The forecast assumes that the pilot
pairs behave like the validation pairs within each grade; the validation sample decides.

## Throughput

At 16 parallel requests the judge graded 5,000–5,600 pairs per hour. The 6,350 pairs of the
grading run would take about 1 h 15 min, plus about 11 minutes of setup. The server's request
queue stayed nearly empty, so more parallel requests would shorten this.

## What this means for the open decisions

- **Answer cap.** 4,096 tokens is ample. No change is needed.
- **Sampling setting (for the advisor to decide).** The author's recommendation is temperature 0.
  It is the more repeatable setting in the pilot, it loses nothing in agreement with the official
  grades, and a grade taken at temperature 0 should agree with a single sampled grade at least as
  often as a second sample does. Against it: the organisers sent no sampling parameter, so
  temperature 0 is not literally their procedure, and it shifts some middle grades to the
  extremes. The evidence for either choice is modest at 270 pairs.
- **Thresholds.** Nothing in the pilot is a reason to change them. They were recorded before any
  judge output and are not edited here.
