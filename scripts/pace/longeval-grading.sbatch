#!/usr/bin/env bash
# E1: grading the judging pool with the local GPT-OSS-120B judge (EXTENSION_PROTOCOL_V1 E1; amendment 5).
#   sbatch longeval-grading-01/scripts/pace/longeval-grading.sbatch
# The serving stack, kernels, model and server settings are the judge pilot's. Two grading folders: the
# order rerun's 22 substitutions first (a live check), then the primary pool and validation sample. No
# analysis runs here: coverage.analyze needs the held-out evidence and runs locally. See LONGEVAL_GRADING.md.
#SBATCH --job-name=longeval-grading
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu,ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --tmp=160G
#SBATCH --time=06:00:00
#SBATCH --output=longeval-grading-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated Epzero checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/longeval-grading-01"
CPU_PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
UV="$ROOT/build/pace-longeval/tools/bin/uv"
OUTPUT="$ROOT/build/pace-longeval-grading/jobs/${SLURM_JOB_ID:?requires Slurm}"
MIN_GPU_MIB="${LONGEVAL_JUDGE_MIN_GPU_MIB:-100000}"
PHASE=initialization
SERVER_PID=
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    printf 'exit_code=%s phase=%s\n' "$code" "$PHASE"
    if [ -n "${STAGE:-}" ]; then
        for name in packages.txt judge-install.json; do
            if [ -f "$STAGE/$name" ]; then cp "$STAGE/$name" "$OUTPUT/" || true; fi
        done
        if [ -f "$STAGE/model/model-identity.json" ]; then cp "$STAGE/model/model-identity.json" "$OUTPUT/" || true; fi
        # The server log can be large; keep its start and its end.
        if [ -f "$STAGE/server.log" ]; then
            head -c 2000000 "$STAGE/server.log" > "$OUTPUT/server-log-head.txt" || true
            tail -c 2000000 "$STAGE/server.log" > "$OUTPUT/server-log-tail.txt" || true
        fi
    fi
    exit "$code"
}
trap finish EXIT

PHASE=package
echo "phase=$PHASE"
(cd "$PACKAGE" && sha256sum --check --quiet --strict SHA256SUMS)
cp "$PACKAGE/SHA256SUMS" "$PACKAGE/manifest.json" "$OUTPUT/"

PHASE=gpu
echo "phase=$PHASE"
# Fail in seconds, before any download: the model needs a large GPU and the pinned stack a CUDA 13 driver.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$OUTPUT/gpu.txt"
MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
[ "$MIB" -ge "$MIN_GPU_MIB" ] || { echo "GPU has ${MIB} MiB; at least ${MIN_GPU_MIB} MiB required" >&2; exit 1; }
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d ' ')"
[ "${DRIVER%%.*}" -ge 580 ] || { echo "NVIDIA driver ${DRIVER}; the pinned CUDA 13 packages need 580 or newer" >&2; exit 1; }

PHASE=environment
echo "phase=$PHASE"
source "$PACKAGE/scripts/pace/longeval-node-env.sh"
STAGE="$LONGEVAL_NODE_STAGE"
/usr/bin/python3 - "$STAGE" <<'PY'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free
if free < 110 * 1024**3:
    raise SystemExit(f'Need at least 110 GiB free on the compute-node temporary disk; found {free / 1024**3:.0f} GiB')
PY
cp -a "$PACKAGE" "$STAGE/package"
PACKAGE="$STAGE/package"
export PYTHONPATH="$PACKAGE" PIP_DISABLE_PIP_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
# Requests to the local server must not go through a site proxy.
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}" NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
{
    printf 'node_stage=%s\n' "$STAGE"
    findmnt -T "$STAGE" -o SOURCE,FSTYPE,TARGET
    df -h "$STAGE"
} > "$OUTPUT/node-storage.txt"

PHASE=tests
echo "phase=$PHASE"
(cd /tmp && "$CPU_PYTHON" -P -m pytest -q -p no:cacheprovider \
    "$PACKAGE/research/longeval/tests/test_judge.py" "$PACKAGE/research/longeval/tests/test_coverage.py" \
    "$PACKAGE/research/longeval/tests/test_grading_inputs.py" --junitxml="$OUTPUT/tests.xml")
# Both grading folders against their contracts (coverage.read_inputs), before any download.
(cd /tmp && "$CPU_PYTHON" -P - "$PACKAGE/inputs/grading" <<'PY') | tee "$OUTPUT/inputs-validation.json"
import json, sys
from pathlib import Path
from research.longeval.coverage import read_inputs
root = Path(sys.argv[1])
print(json.dumps({name: read_inputs(root / name)['pairs'] for name in ('substitutions', 'primary')}))
PY

PHASE=gpu-environment
echo "phase=$PHASE"
"$UV" venv --python "$CPU_PYTHON" --seed "$STAGE/venv"
GPU_PYTHON="$STAGE/venv/bin/python"
"$GPU_PYTHON" -m pip install --only-binary=:all: --report "$STAGE/judge-install.json" \
    -r "$PACKAGE/research/longeval/judge-requirements.txt"
# Prebuilt GPU kernels for the pinned FlashInfer. Without them the server compiles kernels while it
# loads the model, which stopped job 7205655 (no build tool on its search path).
"$GPU_PYTHON" -m pip install --only-binary=:all: --no-deps --require-hashes --timeout 120 --retries 10 \
    -r "$PACKAGE/research/longeval/judge-kernels.txt"
"$GPU_PYTHON" -m pip freeze > "$STAGE/packages.txt"
"$GPU_PYTHON" -m pip check || echo 'warning: pip check reported inconsistencies (recorded in the log)'
# A fallback for any kernel the prebuilt set does not cover: the environment's own build tools
# (ninja and the pinned CUDA compiler) on the search path of the server.
export PATH="$STAGE/venv/bin:$PATH"
# FlashInfer keeps its build workspace under the home directory unless told otherwise; keep it on the node.
export FLASHINFER_WORKSPACE_BASE="$STAGE"
GPU_SITE="$("$GPU_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [ -x "$GPU_SITE/nvidia/cu13/bin/nvcc" ]; then export CUDA_HOME="$GPU_SITE/nvidia/cu13"; fi
# Fail here, before the 65 GB model download, if the kernels are missing or belong to another version.
"$GPU_PYTHON" - <<'PY' | tee "$OUTPUT/kernel-toolchain.json"
import importlib.metadata as metadata
import importlib.util
import json
import os
import shutil


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


wanted = version('flashinfer-python')
kernels = {name: version(name) for name in ('flashinfer-cubin', 'flashinfer-jit-cache', 'flashinfer-jit-cache-sm90a')}
report = {'flashinfer_python': wanted, 'kernels': kernels,
          'cubin_module_found': importlib.util.find_spec('flashinfer_cubin') is not None,
          'cuda_home': os.environ.get('CUDA_HOME'),
          'tools': {name: shutil.which(name) for name in ('ninja', 'c++', 'g++', 'gcc')}}
home = report['cuda_home']
report['tools']['nvcc'] = shutil.which(os.path.join(home, 'bin', 'nvcc')) if home else shutil.which('nvcc')
print(json.dumps(report, indent=2))
problems = [f'{name} is {found}, expected {wanted}' for name, found in kernels.items()
            if found is None or found.split('+')[0] != wanted]
if not report['cubin_module_found']:
    problems.append('the flashinfer_cubin module cannot be found')
if report['tools']['ninja'] is None:
    problems.append('ninja is not on the search path')
if problems:
    raise SystemExit('prebuilt kernels are not usable: ' + '; '.join(problems))
PY

PHASE=model
echo "phase=$PHASE"
(cd /tmp && "$CPU_PYTHON" -P -m research.longeval.judge fetch-model \
    "$PACKAGE/research/longeval/judge-model.json" "$STAGE/model")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
{
    date -u
    hostname
    nvidia-smi
} > "$OUTPUT/hardware.txt"

PHASE=server
echo "phase=$PHASE"
PORT="$(/usr/bin/python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
BASE_URL="http://127.0.0.1:${PORT}/v1"
cp "$PACKAGE/scripts/pace/longeval-judge-server.yaml" "$OUTPUT/server-config.yaml"
# The server runs as a job step in the background; the exit trap stops it.
srun "$STAGE/venv/bin/vllm" serve "$STAGE/model" \
    --config "$PACKAGE/scripts/pace/longeval-judge-server.yaml" \
    --served-model-name openai/gpt-oss-120b --host 127.0.0.1 --port "$PORT" > "$STAGE/server.log" 2>&1 &
SERVER_PID=$!
(cd /tmp && "$CPU_PYTHON" -P -m research.longeval.judge wait-server --base-url "$BASE_URL" --pid "$SERVER_PID" --timeout 5400) \
    | tee "$OUTPUT/server-ready.json"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee "$OUTPUT/gpu-memory-loaded.txt"

PHASE=grade-substitutions
echo "phase=$PHASE"
# Temperature 0 (decision 1); the server settings are bound by checksum.
(cd /tmp && "$CPU_PYTHON" -P -m research.longeval.coverage grade "$PACKAGE/inputs/grading/substitutions" \
    "$OUTPUT/grading/substitutions" --base-url "$BASE_URL" --temperature 0 \
    --server-config "$PACKAGE/scripts/pace/longeval-judge-server.yaml" --workers 16) | tee "$OUTPUT/substitutions-summary.json"
# A live check: every one of these answers must parse before the primary pool is graded.
"$CPU_PYTHON" - "$OUTPUT/grading/substitutions/grading.json" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
if record['pairs_without_a_usable_answer']:
    raise SystemExit(f"{record['pairs_without_a_usable_answer']} substitution pairs got no usable answer")
PY

PHASE=grade-primary
echo "phase=$PHASE"
(cd /tmp && "$CPU_PYTHON" -P -m research.longeval.coverage grade "$PACKAGE/inputs/grading/primary" \
    "$OUTPUT/grading/primary" --base-url "$BASE_URL" --temperature 0 \
    --server-config "$PACKAGE/scripts/pace/longeval-judge-server.yaml" --workers 16) | tee "$OUTPUT/primary-summary.json"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee "$OUTPUT/gpu-memory-final.txt"
(cd "$OUTPUT/grading" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
PHASE=done
printf 'PASS: grading completed; analyse locally with coverage analyze: %s\n' "$OUTPUT"
