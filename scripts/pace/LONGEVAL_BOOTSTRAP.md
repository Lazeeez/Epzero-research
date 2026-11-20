# PACE bootstrap after the source transfer

The account owner verified the original 123-file source overlay in the isolated scratch checkout
and reported the old experiment jobs cancelled on 2026-11-16. The pasted preflight confirms account
`coc`, QOS `coc-ice`, and Gold 6226 nodes in `coc-cpu,ice-cpu`. Direct agent SSH authentication is
still unavailable; submission is performed by the account owner. This supersedes the earlier
preflight status in LONGEVAL.md. No successful PACE build or submission is claimed yet.

Overlay the separately checksummed bootstrap supplement, then run from the checkout root:

```bash
bash scripts/pace/longeval-setup.sh
sbatch scripts/pace/longeval-bootstrap.sbatch
```

Setup keeps Python, caches and wheels under `build/pace-longeval` on scratch. It downloads binary
wheels and GoogleTest sources, records their SHA-256 identities, and verifies both source manifests.
GoogleTest initially uses HTTPS release provenance; the recorded checksum binds the subsequent
offline build, not an independently authenticated upstream release. The wheel receipt and final
package freeze record resolved transitive/build dependencies. Setup can resume downloads; if
dependency resolution changes between attempts, use a fresh stage rather than mix wheel versions.

The Slurm job uses four CPUs and 16 GB for up to one hour on a shared node. GCC 13.3.0 follows the
existing PACE toolchain recipe; its module must be available. It installs the staged wheels offline,
compiles this checkout, runs C++ and research/core Python tests, and verifies the synthetic pilot.
Job-specific evidence is in `build/pace-longeval/jobs/JOB_ID/`; `exit-code.txt` must be zero and the
Slurm log must end in `PASS`. Setup and builds share a lock to prevent concurrent environment writes.
This job measures correctness, not isolated latency, full-corpus relevance, or novelty.

After a successful build, CPU research jobs use:

```bash
export LONGEVAL_PYTHON="$PWD/build/pace-longeval/venv/bin/python"
```

Do not stage the entire corpus based on filesystem free space alone: the preflight did not establish
the account's scratch quota. Encoder agreement and real-query validation remain separate research
gates. GPU packages and model files are not included in this CPU bootstrap.
