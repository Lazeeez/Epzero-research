# Local validation — 2026-11-16

This records pipeline validation, not retrieval research results. The input is entirely synthetic.

Subsequent work resolved the native-build limitation through Ubuntu WSL; see
[NATIVE_VALIDATION.md](NATIVE_VALIDATION.md). This earlier execution record is retained unchanged
as evidence of what was tested at that stage.

## Environment and provenance

- Checkout: `ce951f6b5176527597f172e18003ffc87713952f`, with the research foundation and IR changes
  uncommitted. Source files, including local changes, are hashed in the run manifest.
- Harness/source digest: `2edb8cd288f1ecee53f43ad4311539c5255d37f18b5835f655f93096ecc73a39`.
- Windows 10.0.26300, Python 3.11.16, NumPy 2.4.6, FAISS CPU 1.16.0, Epzero 0.5.0,
  psutil 7.2.2; independent evaluator pytrec-eval-terrier 0.5.11.
- Native binary origin: **released wheel**. Binary SHA-256:
  `1e4af7bc9a920e08c242ad7c1a313ffba428285a9a5fbb27179f91a8a38e71fd`.
- A source build was attempted with MSVC 14.44 and CMake 4.4.3. It could not link the compiler probe
  because the Windows SDK (`kernel32.lib`) is missing. No native source-build or sanitizer pass is
  claimed. No C++ engine code was changed.

## Test execution

48 tests passed, zero failed/skipped: 34 research foundation tests and 14 existing/new IR tests.
The modified `engine/python/epzero/ir.py` was loaded from the checkout, while its native dependency
came from the released wheel. The local invocation explicitly selected that source module:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& build/longeval-venv/Scripts/python.exe -c 'import pathlib, epzero, pytest; epzero.__path__.insert(0, str(pathlib.Path("engine/python/epzero").resolve())); from epzero import ir; print("Testing source:", ir.__file__); raise SystemExit(pytest.main(["-q", "research/longeval/tests", "engine/python/tests/test_ir.py", "--basetemp=build/longeval-test-02", "--tb=short"]))'
```

Use a new basetemp when repeating this command. Validation includes independently computed metric
cases, TREC evaluator parity, independent exhaustive scorers, query leakage, invalid snapshot/output
rejection, calibration failure/nonmonotonicity, the full trajectory, retained failed runs, and artifact
corruption. It is not the entire engine/integration test suite.

## Pilot and fresh-process replay

Executed fixture generation and two separate CLI runs using the supplied development configuration:

```powershell
& build/longeval-venv/Scripts/python.exe -m research.longeval fixture build/longeval-fixture
& build/longeval-venv/Scripts/python.exe -m research.longeval run --dataset build/longeval-fixture/dataset.json --output build/longeval-pilot --native-origin released-wheel
& build/longeval-venv/Scripts/python.exe -m research.longeval verify build/longeval-pilot --dataset build/longeval-fixture/dataset.json
& build/longeval-venv/Scripts/python.exe -m research.longeval run --dataset build/longeval-fixture/dataset.json --output build/longeval-replay --native-origin released-wheel
& build/longeval-venv/Scripts/python.exe -m research.longeval verify build/longeval-replay --dataset build/longeval-fixture/dataset.json
```

Both runs completed. Each produced 153 ranked run files, 72 state records, and 804 component-cost
records; the verifier checked 157 artifact files and re-scored every result row. All 153 run files,
the state fingerprints, and provenance matched between processes. Timing measurements are expected
to differ and were not compared for equality. The 72 calibration selections met the fixture target;
this is not a claim about real-data target attainment.

The local artifacts are in `build/longeval-pilot/` and `build/longeval-replay/` (ignored build outputs).
The first manifest SHA-256 is
`4ced82ad9de839da4457d900737a5c1d807d69795400e5b23b8687b4bc4a52b6`.
Regenerate them from the commands above in fresh directories; future source changes will correctly
produce different provenance hashes. This record is a dated observation, not a permanent green gate.

`git diff --check` passed. A Linux/Windows source-build CI workflow is supplied, but it has not been
executed on the remote CI service in this session. No real LongEval corpus, embeddings, or qrels were
downloaded or used. See STATUS.md for the remaining G0–G3 work.
