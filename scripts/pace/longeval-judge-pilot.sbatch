#!/usr/bin/env bash
# Judge pilot (exploratory follow-up): GPT-OSS-120B on prompt-development pairs. See LONGEVAL_JUDGE_PILOT.md.
#SBATCH --job-name=longeval-judge
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu,ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --tmp=160G
#SBATCH --time=04:00:00
#SBATCH --output=longeval-judge-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated Epzero checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/longeval-judge-pilot-01"
CPU_PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
UV="$ROOT/build/pace-longeval/tools/bin/uv"
OUTPUT="$ROOT/build/pace-longeval-judge-pilot/jobs/${SLURM_JOB_ID:?requires Slurm}"
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
    "$PACKAGE/research/longeval/tests/test_judge.py" --junitxml="$OUTPUT/tests.xml")
(cd /tmp && "$CPU_PYTHON" -P -m research.longeval.judge validate "$PACKAGE/inputs") | tee "$OUTPUT/inputs-validation.json"

PHASE=gpu-environment
echo "phase=$PHASE"
"$UV" venv --python "$CPU_PYTHON" --seed "$STAGE/venv"
GPU_PYTHON="$STAGE/venv/bin/python"
"$GPU_PYTHON" -m pip install --only-binary=:all: --report "$STAGE/judge-install.json" \
    -r "$PACKAGE/research/longeval/judge-requirements.txt"
"$GPU_PYTHON" -m pip freeze > "$STAGE/packages.txt"
"$GPU_PYTHON" -m pip check || echo 'warning: pip check reported inconsistencies (recorded in the log)'

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
srun --chdir="$STAGE" "$STAGE/venv/bin/vllm" serve "$STAGE/model" \
    --config "$PACKAGE/scripts/pace/longeval-judge-server.yaml" \
    --served-model-name openai/gpt-oss-120b --host 127.0.0.1 --port "$PORT" > "$STAGE/server.log" 2>&1 &
SERVER_PID=$!
(cd /tmp && "$CPU_PYTHON" -P -m research.longeval.judge wait-server --base-url "$BASE_URL" --pid "$SERVER_PID" --timeout 5400) \
    | tee "$OUTPUT/server-ready.json"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee "$OUTPUT/gpu-memory-loaded.txt"

PHASE=probe
echo "phase=$PHASE"
(cd /tmp && "$CPU_PYTHON" -P -m research.longeval.judge probe "$PACKAGE/inputs" "$OUTPUT/judging" --base-url "$BASE_URL") \
    | tee "$OUTPUT/probe-summary.json"

PHASE=judge
echo "phase=$PHASE"
(cd /tmp && "$CPU_PYTHON" -P -m research.longeval.judge judge "$PACKAGE/inputs" "$OUTPUT/judging" \
    --base-url "$BASE_URL" --workers 16)

PHASE=analyze
echo "phase=$PHASE"
(cd /tmp && "$CPU_PYTHON" -P -m research.longeval.judge analyze "$PACKAGE/inputs" "$OUTPUT/judging") \
    | tee "$OUTPUT/analysis-summary.json"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee "$OUTPUT/gpu-memory-final.txt"
(cd "$OUTPUT/judging" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
PHASE=done
printf 'PASS: judge pilot completed and summarised: %s\n' "$OUTPUT"
