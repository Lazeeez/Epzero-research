#!/usr/bin/env bash
# Login-node preparation: download Python, binary wheels and test sources only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="$PWD"
STAGE="$ROOT/build/pace-longeval"
mkdir -p "$STAGE"
exec 9>"$STAGE/setup.lock"
flock -n 9 || { echo 'Another setup/build holds the research environment lock' >&2; exit 1; }
/usr/bin/python3 scripts/pace/longeval-stage.py source
if [ -f "$STAGE/staged-files.json" ]; then
    /usr/bin/python3 scripts/pace/longeval-stage.py verify
    echo 'Existing staging verified. Submit scripts/pace/longeval-bootstrap.sbatch.'
    exit 0
fi
export PIP_CACHE_DIR="$STAGE/pip-cache" UV_CACHE_DIR="$STAGE/uv-cache"
export UV_PYTHON_INSTALL_DIR="$STAGE/python" UV_PYTHON_BIN_DIR="$STAGE/bin"
export PIP_DISABLE_PIP_VERSION_CHECK=1
if [ ! -x "$STAGE/tools/bin/uv" ]; then
    /usr/bin/python3 -m venv "$STAGE/tools"
    "$STAGE/tools/bin/python" -m pip install --only-binary=:all: 'uv>=0.9,<1'
fi
if [ ! -x "$STAGE/venv/bin/python" ]; then
    "$STAGE/tools/bin/uv" venv --python 3.12 --seed "$STAGE/venv"
fi
WHEELS="$(mktemp -d "$STAGE/wheels.XXXXXX")"
"$STAGE/venv/bin/python" -m pip download --only-binary=:all: --dest "$WHEELS" \
    -r research/longeval/requirements.txt 'scikit-build-core==1.1.2' 'pybind11==3.1.0' 'cmake>=3.28,<5' ninja
if [ -d "$STAGE/wheelhouse" ]; then
    mv "$STAGE/wheelhouse" "$STAGE/wheelhouse-previous-$(date +%s)"
fi
mv "$WHEELS" "$STAGE/wheelhouse"
"$STAGE/venv/bin/python" scripts/pace/longeval-stage.py stage
{
    date -u
    "$STAGE/tools/bin/uv" --version
    "$STAGE/venv/bin/python" --version
    sha256sum longeval-source-manifest.json longeval-bootstrap-manifest.json "$STAGE/staged-files.json"
} > "$STAGE/setup-receipt.txt"
echo 'Staging complete. Submit scripts/pace/longeval-bootstrap.sbatch from this checkout.'
