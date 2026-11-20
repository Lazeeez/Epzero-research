#!/usr/bin/env bash
# One-command check of the research code; needs no LongEval data and takes a few minutes.
#   python -m pip install -r research/longeval/requirements.txt
#   bash scripts/verify.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON="${PYTHON:-python}"

echo "== 1/3 unit and regression tests"
"$PYTHON" -m pytest -q -p no:cacheprovider research/longeval/tests scripts/pace/tests

echo "== 2/3 frozen protocol integrity"
"$PYTHON" scripts/verify_freeze.py

echo "== 3/3 synthetic three-snapshot pilot (build, search, score, verify)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
"$PYTHON" -m research.longeval fixture "$OUT/fixture" > /dev/null
"$PYTHON" -m research.longeval run --dataset "$OUT/fixture/dataset.json" --output "$OUT/pilot" --native-origin released-wheel
"$PYTHON" -m research.longeval verify "$OUT/pilot" --dataset "$OUT/fixture/dataset.json"
echo "== all checks passed"
