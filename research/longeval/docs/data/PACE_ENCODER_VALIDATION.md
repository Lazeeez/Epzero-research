# PACE encoder diagnostic — job 7106389

The account owner supplied the final job log. It reports both precision diagnostics completed,
all 127 original source files and 12 GPU-bundle files verified, and the final `PASS` message.
The full result archive has now been retrieved and independently verified locally.

The pasted float32/CUDA comparison covers all 32 hash-selected first-snapshot documents:

| Statistic | GPU float32 | GPU BF16 | CPU BF16 |
|---|---:|---:|---:|
| Minimum cosine similarity | 0.9998533063724798 | 0.999659054125649 | 0.9996131269767212 |
| Median cosine similarity | 0.9999491416916138 | 0.9998033661538943 | 0.9997883869526796 |
| Maximum cosine similarity | 0.9999701911930063 | 0.9998532929990918 | 0.9998640770392512 |
| Bitwise-identical rows | 0 / 32 | 0 / 32 | 0 / 32 |
| Maximum absolute coordinate error | 0.0016033239662647247 | 0.003955204039812088 | 0.0031345579773187637 |

Both comparisons identify the same reference manifest SHA-256:
`aaa62e811fede1dee0c92f7658c533f5263e32730ac774349f3636ebd679eea6`.
The GPU float32 encoding report SHA-256 printed in the log is
`32002f8a4cb6ad7478e209f934667076cdafe624056a1b385efdf4e9a95d8560`.
The comparison source SHA-256 is
`bca16d4a279a26bd85288ff2e1ff2ea66466ee7a2f71388cdc664548cd2a209b`.

Within the same GPU workflow, float32 cosine agreement exceeds BF16 agreement on all 32 sampled
documents. Model identities, input texts, token counts, pooling, normalization, attention,
truncation and maximum length match; both GPU runs disable TF32. This supports testing float32
as a candidate reconstruction for the next training-query diagnostic, while retaining BF16 as
a sensitivity comparison. It does not freeze the final research representation.
No acceptance threshold, historical-runtime reproduction, or query compatibility is established
by these document-sample results. Query encodings and retrieval-sensitivity diagnostics remain
pending; the job produced no research query embeddings.

The Slurm epilog reports account `coc`, QOS `coc-ice`, partition `coc-gpu`, node
a single compute node, four CPUs, one A100 and 40 GB host RAM. Wall time was 00:05:42;
reported memory was `16316052K`. This is whole-job scheduler accounting, not GPU memory
or an isolated model-inference benchmark. The timestamp `Nov-16-2026 01:48:27` has no
timezone attached in the supplied log.

Persistent result directory:

```text
<PACE checkout>/build/pace-longeval-encoder/jobs/7106389
```

Retrieve that result directory and `longeval-agreement-7106389.log` only. It contains small
diagnostic arrays, receipts and reports. Large model/package/environment downloads used the
compute-node temporary storage workflow; do not copy or re-download them to shared scratch.

## Independent verification and storage evidence

Returned archive `build/longeval-agreement-7106389-evidence.tar.gz`, SHA-256
`7cb93e9c9f640f9dcdd6b9a631d55b626ee334c047da6c1a3cf4cc80b3dc04d3`, contains 630,015 bytes.
It was safely extracted under `build/longeval-pace-encoder-evidence-7106389/`.
The exit-code file is zero, the GPU manifest matches the previously verified agreement-03
bundle, and all input/vector/ID/model-report identities match. Local recomputation reproduces
every reported per-document measurement within absolute tolerance `1e-14`.
The machine-readable result is `local-verification.json` in that extracted directory.

The storage report locates the job stage at `/tmp/longeval-7106389.ctks6E`, on local XFS mounts
backed by `/dev/mapper/sysvg-tmp` and `/dev/mapper/sysvg-scratch`. The physical scratch device
name here refers to the compute-node local disk, not the shared user scratch directory.
The recorded GPU is NVIDIA A100-PCIE-40GB. The installed model identity matches the CPU run.

Comparing the reconstructed encoders directly gives median cosine similarity
`0.9997616901791222` for CPU BF16 versus GPU BF16, and `0.999838730500386` for GPU BF16 versus
GPU float32. Thus CPU/GPU BF16 outputs are not interchangeable at the bit level. Keep device,
precision and runtime fixed within each research trajectory. The next validation should encode
the released training queries with the exact baseline prompt in both GPU precisions, then measure
top-100 ranking sensitivity against the unchanged archived vectors before primary experiments.
