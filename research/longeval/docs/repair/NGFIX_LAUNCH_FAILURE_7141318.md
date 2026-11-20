# NGFix launcher correction v03

PACE job7141318 ended FAILED, exit1:0, elapsed2s. The author subsequently provided its
log, confirming `No module named research.longeval.ngfix_screen` at the first
Python module invocation. This matches the independently reproduced launch-path
defect below. The original log file has not been copied locally; the pasted evidence
is recorded in `build/longeval-ngfix-7141318-confirmed-failure.json`.

A real v02 launch defect was independently reproduced using the archived native-cost
checkout as the working directory. The first `python -m research.longeval.ngfix_screen`
command imports that checkout's older `research.longeval` package before the isolated
supplement on PYTHONPATH, and exits with `No module named research.longeval.ngfix_screen`.
The earlier package checks ran inside the new supplement and missed this launch context.

V03 uses Python `-P` to omit the implicit working-directory import path, while retaining
the explicit supplement PYTHONPATH. It adds phase logging to identify setup failures.
All native sources, input artifacts, compiler flags and storage controls are byte-identical
to the compile-checked v02 package. Only the launcher and its instructions changed;
the package adds a regression test and the derived-package builder.

Corrected archive: `build/longeval-ngfix-screen-03.tar.gz`, 2,546,174bytes, SHA-256:
`9be895484ad89c6246fc49346254609328aabadbb51e754b9c91c6e2c26e90d3`.
All500 packaged files verify; launch from the historical checkout succeeds; shell
syntax passes; all22 packaged tests pass. Native AVX-512 execution remains pending.

The failure occurred before compilation, synthetic checks and corpus downloads.
Keep the failed log as evidence; this was a launcher failure, not an algorithm result.

For the corrected version, copy the v03 archive into the same PACE checkout:

```bash
echo '9be895484ad89c6246fc49346254609328aabadbb51e754b9c91c6e2c26e90d3  longeval-ngfix-screen-03.tar.gz' | sha256sum -c -
tar -xzf longeval-ngfix-screen-03.tar.gz
PYTHONPATH="$PWD/longeval-ngfix-screen-03" build/pace-longeval/venv/bin/python -P \
  -m research.longeval.ngfix_screen package-verify "$PWD/longeval-ngfix-screen-03"
sbatch longeval-ngfix-screen-03/scripts/pace/longeval-ngfix.sbatch
```

The v03 package includes complete retrieval instructions in `scripts/pace/LONGEVAL_NGFIX.md`.
All large data remains on compute-node temporary disk. No resubmission had been
performed at the time, and no NGFix runtime result is claimed.
