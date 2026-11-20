# Epzero research

This repository holds the research built around [Epzero](https://github.com/Lazeeez/Epzero), an
in-memory vector search engine. The engine, its benchmarks and the website live in the Epzero
repository; this one contains the studies, their code, protocols, results and literature reviews.

## Contents

| Path | Contents |
| --- | --- |
| [`research/longeval/`](research/longeval) | **The LongEval study**: does an approximate nearest-neighbour index change dense-retrieval conclusions as a collection grows? Code, tests, the frozen protocol, results and the CLEF 2027 manuscript draft. Start with its [README](research/longeval/README.md). |
| [`scripts/pace/`](scripts/pace) | PACE cluster runbooks, job scripts and packagers for the LongEval study, indexed in [LONGEVAL.md](scripts/pace/LONGEVAL.md) |
| [`docs/research-plan.md`](docs/research-plan.md) | Research plan for an experimental study of in-memory graph search |
| [`reports/`](reports) | Literature reviews, the LongEval novelty review and the LongEval statement of research |
| [`research_notes/`](research_notes) | Source notes behind each literature review, one folder per report |

## Running the LongEval code

The research code uses the released Epzero engine from PyPI. With Python 3.11 or later:

```sh
python -m pip install -r research/longeval/requirements.txt
python -m pytest -q research/longeval/tests
```

Tests marked `linux_native` run only on Linux, and the `hnswlib` comparison runs only where that
library is installed. Full experiments run on the PACE cluster; see
[`scripts/pace/LONGEVAL.md`](scripts/pace/LONGEVAL.md).

## Data policy

LongEval data are not redistributed: document vectors, query texts and vectors, judgments, and run
files. They are downloaded from the pinned official release and kept under `build/`, which git
ignores. The repository holds code, configurations, protocols, aggregate results and figures only.

## License

MIT; see [LICENSE](LICENSE).
