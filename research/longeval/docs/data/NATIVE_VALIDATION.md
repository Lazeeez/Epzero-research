# Native source validation — 2026-11-16

The native source-build limitation in the earlier [Windows validation](VALIDATION.md) is resolved
using the existing Ubuntu WSL installation. The missing Windows SDK was not installed and no global
toolchain or memory configuration was changed.

## Build and tests

- Ubuntu GCC 13.3.0, Python 3.12.3, Linux 6.6.87.2 WSL2, glibc 2.39.
- A workspace-local environment at `build/longeval-linux-venv/` installed Epzero from `./engine`.
- **115 / 115 C++ tests passed** in a fresh Release build with GCC and Ninja.
- **106 Python tests passed**, zero failed/skipped: 48 research/audit tests and the four core
  `test_ir`, `test_epzero`, `test_auto`, and `test_metadata` suites.
- The new download/audit tests also passed on Windows (14 / 14).
- No sanitizer or full framework-integration-suite pass is claimed. Remote CI has not run.

Commands below run from the repository root inside Ubuntu. WSL callers can prefix them with
`wsl -d Ubuntu --`; use fresh pytest/output directories when repeating.

```sh
python3 -m venv build/longeval-linux-venv
build/longeval-linux-venv/bin/python -m pip install -r research/longeval/requirements.txt scikit-build-core pybind11
CMAKE_BUILD_PARALLEL_LEVEL=2 build/longeval-linux-venv/bin/python -m pip install --no-build-isolation --no-deps ./engine
build/longeval-linux-venv/bin/python -m pytest -q research/longeval/tests engine/python/tests/test_ir.py engine/python/tests/test_epzero.py engine/python/tests/test_auto.py engine/python/tests/test_metadata.py --basetemp=build/longeval-linux-test-02 --junitxml=build/longeval-linux-test-02.xml
cmake -S engine -B build/longeval-cpp -G Ninja -DCMAKE_BUILD_TYPE=Release -DEPZERO_BUILD_TESTS=ON -DEPZERO_BUILD_BENCH=OFF -DEPZERO_BUILD_NODE=OFF
cmake --build build/longeval-cpp --parallel 2
ctest --test-dir build/longeval-cpp --output-on-failure
```

Single-thread OMP/OpenBLAS/MKL environment variables were set before tests and pilots. Detailed
native test logs remain under `build/longeval-cpp/Testing/Temporary/`. The Python JUnit report is
`build/longeval-linux-test-02.xml`, SHA-256
`9d9cf7adfaa2b5c24a7273d738636f71d6ce25e0aabefc40e98ae3eb466e6f7e`.

## Fresh-process synthetic replay

Two CLI runs used this newly compiled engine, the same fixture, and the three-seed configuration:

```sh
build/longeval-linux-venv/bin/python -m research.longeval run --dataset build/longeval-fixture/dataset.json --output build/longeval-native-pilot --native-origin checkout-build
build/longeval-linux-venv/bin/python -m research.longeval verify build/longeval-native-pilot --dataset build/longeval-fixture/dataset.json
build/longeval-linux-venv/bin/python -m research.longeval run --dataset build/longeval-fixture/dataset.json --output build/longeval-native-replay --native-origin checkout-build
build/longeval-linux-venv/bin/python -m research.longeval verify build/longeval-native-replay --dataset build/longeval-fixture/dataset.json
```

Both completed, each with 153 rankings and 157 verified artifacts. Every ranking file and the state
fingerprints matched between the separate processes. Timings are not expected to match. No
cross-platform bitwise-reproducibility claim is made.

Native binary SHA-256: `01bff2709e8dabfb2ad380861120f3a3b603199e984c8b9d3957eee2b17cc402`.
Source digest: `fc50f5bd4fd7e4d2a9d0fe9e120917d2b91ae87467bf0eac8078d0e720b53cde`.
The checkout remains `ce951f6b5176527597f172e18003ffc87713952f` plus the recorded local changes.

| Manifest | SHA-256 |
|---|---|
| `build/longeval-native-pilot/manifest.json` | `acbd376b08c2e633d6bcd48da8bf1e1b3e983a1d82533c1cb7b6311fd3c0f45a` |
| `build/longeval-native-replay/manifest.json` | `04633013691ec36988f2ff29e9acdb1b524a46e43081b6c4333f5601effb44c9` |

This is native correctness and synthetic-pipeline evidence. Native checkpoint continuation is still
disabled by the research protocol, and neither this test suite nor the real-file audit establishes
an ANN-history relevance effect, a full-data resource budget, or novelty.
