# Audited document agreement on PACE

The additive GPU bundle leaves the original 127 validated source files unchanged. Its new
manifest binds the comparison implementation, tests, job scripts and a 32-document reference
sample. The sample was selected by document-ID hash across all 869,902 eligible first-snapshot
documents; no judgments or encoder outputs enter selection. The original archive and full
vector file were verified before extracting the small reference bundle.

After extracting the supplemental archive in the existing scratch checkout:

```bash
sbatch scripts/pace/longeval-agreement.sbatch
```

The job itself calls setup inside its compute allocation. Do not run setup on login: it now
requires the compute-node Slurm environment and refuses shared/network temporary storage.
The pinned 8.04 GB model, CUDA packages, Python environment, installation temporary files and
all configured caches live in a job-specific directory under the compute node's temporary
directory. Only the small reference bundle, source, logs, receipts and diagnostic outputs remain
in the existing shared checkout. No new large downloads go to shared scratch or home.

PACE documents `${TMPDIR}` as the job's local disk and `#SBATCH --tmp` as the local-space request
in its [orientation guide](https://docs.pace.gatech.edu/training/img/PACE_orientation_Aug2023.pdf).
The script accepts `${SLURM_TMPDIR}` when provided, otherwise `${TMPDIR}`, and checks the resolved
filesystem before creating its directory. It requests 80 GB of local disk and checks at least
40 GiB is free before download. There is no shared-storage fallback. PACE removes its job-local
temporary directory after the allocation; the next job must download/install again. The script
does not recursively delete any existing directory.

Compute-node outbound access has not yet been tested in this workflow. A download failure
fails the job and leaves its small receipts/logs in the output folder; it does not redirect
downloads to shared scratch. Existing scratch-resident bootstrap environments are read only.

Job 7104607 stopped before downloads because `findmnt` returned two `xfs` records and the
original guard treated them as one filesystem name. Bundle agreement-03 checks every returned
record individually. Repeated local records pass; any shared, memory-backed, unknown or empty
result fails. Eight storage-routing tests, including the exact observed output, pass locally.

Torch 2.9.0 uses the CUDA 12.8 wheel from the
[official release installation instructions](https://pytorch.org/get-started/previous-versions/#v2-9-0).
The compute job checks CUDA and BF16 availability; incompatibility fails without a CPU fallback.
It requests one A100, four CPUs, 40 GB host RAM and four hours (including downloads) using the confirmed account/QOS.
The supplied inventory lists A100 40/80 GB nodes in the selected partitions.

The job encodes the same sample in BF16 and float32 in separate processes, with TF32 disabled,
no truncation, and identical pooling/normalization. Both outputs are compared by document ID to
the original archived float32 vectors. This isolates a precision comparison within this GPU
implementation; it does not reconstruct the historical vLLM runtime. Local CPU BF16 results
provide a separate implementation/device check on the same sample.

Success means the diagnostics ran and their inputs/outputs verified. It does not automatically
establish sufficient agreement: no acceptance threshold is chosen after observing results.
Examine both `*-agreement.json` files and the per-document errors. Query encoding remains a
separate gate. Results are in `build/pace-longeval-encoder/jobs/JOB_ID/`, with Slurm output in
`longeval-agreement-JOB_ID.log`.

The checked baseline query formatting is exactly:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:{original query text}
```

There is no added space after `Query:`. The evidence is the cached baseline at upstream commit
`a488f50f2132f38c07631f484eec0b67f3a18926`; matching this formatting alone does not establish
historical model/runtime identity. Do not substitute a generic model-card prompt or normalize
the original query text silently.
