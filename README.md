# Epzero research

This repository holds the research built around [Epzero](https://github.com/Lazeeez/Epzero), an
in-memory vector search engine. The engine, its benchmarks and the website live in the Epzero
repository; this one contains the studies, their code, protocols, results and literature reviews.

## Contents

| Path | Contents |
| --- | --- |
| [`research/longeval/`](research/longeval) | **The LongEval study**: does an approximate nearest-neighbour index change dense-retrieval conclusions as a collection grows? Code, tests, the frozen protocol, results and the CLEF 2027 manuscript draft. Start with its [README](research/longeval/README.md). |
| [`scripts/`](scripts) | `verify.sh` (the one-command check), and the PACE cluster runbooks, job scripts and packagers, indexed in [scripts/pace/LONGEVAL.md](scripts/pace/LONGEVAL.md) |
| [`REPRODUCE.md`](REPRODUCE.md) | How to reproduce the study, from a three-minute check to every result |
| [`docs/research-plan.md`](docs/research-plan.md) | Research plan for an experimental study of in-memory graph search |
| [`reports/`](reports) | Literature reviews, the LongEval novelty review and the LongEval statement of research |
| [`research_notes/`](research_notes) | Source notes behind each literature review, one folder per report |

## Check it yourself

A one-command check needs no LongEval data and takes about three minutes. It runs the tests,
verifies the frozen protocol, and runs the pipeline end to end on synthetic data:

```sh
python -m venv .venv && source .venv/bin/activate
python -m pip install -r research/longeval/requirements.txt   # includes the released Epzero engine
bash scripts/verify.sh
```

[REPRODUCE.md](REPRODUCE.md) goes further, up to recomputing every result from the official
LongEval release. It gives the hardware and time needed at each level.

## Data policy

LongEval data are not redistributed: document vectors, query texts and vectors, judgments, and run
files. They are downloaded from the pinned official release and kept under `build/`, which git
ignores. The repository holds code, configurations, protocols, aggregate results and figures only.

## License

MIT; see [LICENSE](LICENSE).
