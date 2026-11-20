#!/usr/bin/env bash
# Called by the GPU job: all downloads and installation live on node-local temporary disk.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="$PWD"
source scripts/pace/longeval-node-env.sh
STAGE="$LONGEVAL_NODE_STAGE"
CPU_PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
exec 9>"$STAGE/setup.lock"
flock -n 9 || { echo 'Encoder environment is in use' >&2; exit 1; }
"$CPU_PYTHON" scripts/pace/longeval-gpu-verify.py
export PIP_DISABLE_PIP_VERSION_CHECK=1
if [ ! -x "$STAGE/venv/bin/python" ]; then
    "$ROOT/build/pace-longeval/tools/bin/uv" venv --python "$CPU_PYTHON" --seed "$STAGE/venv"
fi
"$STAGE/venv/bin/python" -m pip install --only-binary=:all: --report "$STAGE/torch-install.json" \
    torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
"$STAGE/venv/bin/python" -m pip install --only-binary=:all: --report "$STAGE/encoder-install.json" \
    -r research/longeval/encoder-requirements.txt
"$STAGE/venv/bin/python" -m pip check
"$STAGE/venv/bin/python" -m pip freeze > "$STAGE/packages.txt"
"$CPU_PYTHON" -m research.longeval fetch-encoder "$STAGE/model"
echo 'Node-local encoder staging complete; the GPU job can now run inference.'
