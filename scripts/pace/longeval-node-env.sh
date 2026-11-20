#!/usr/bin/env bash
# Source only inside a compute allocation. Never fall back to shared storage.
: "${SLURM_JOB_ID:?requires a Slurm compute job}"
: "${SLURMD_NODENAME:?requires the compute-node Slurm environment}"
LONGEVAL_TEMP_ROOT="${SLURM_TMPDIR:-${LONGEVAL_NODE_TMP_ROOT:-${TMPDIR:-}}}"
: "${LONGEVAL_TEMP_ROOT:?Slurm did not provide a node temporary directory}"
LONGEVAL_TEMP_ROOT="$(realpath -e "$LONGEVAL_TEMP_ROOT")"
[ -d "$LONGEVAL_TEMP_ROOT" ] && [ -w "$LONGEVAL_TEMP_ROOT" ] || {
    echo 'Node temporary directory is not writable' >&2; return 1;
}
LONGEVAL_TEMP_FS="$(findmnt -rn -o FSTYPE -T "$LONGEVAL_TEMP_ROOT")"
# PACE can report multiple mount records for one target (observed: xfs followed by xfs).
# Check every record; selecting only the first could hide a shared filesystem result.
if [ -z "$LONGEVAL_TEMP_FS" ]; then
    echo 'Refusing downloads: temporary filesystem could not be determined' >&2; return 1
fi
while IFS= read -r LONGEVAL_FS_RECORD; do
    case "$LONGEVAL_FS_RECORD" in
        xfs|ext4|ext3|btrfs|zfs) ;;
        *) echo "Refusing downloads: temporary directory is not a recognized local disk filesystem ($LONGEVAL_FS_RECORD)" >&2; return 1 ;;
    esac
done <<< "$LONGEVAL_TEMP_FS"
if [ -z "${LONGEVAL_NODE_STAGE:-}" ]; then
    /usr/bin/python3 - "$LONGEVAL_TEMP_ROOT" <<'PY'
import shutil, sys
if shutil.disk_usage(sys.argv[1]).free < 40 * 1024**3:
    raise SystemExit('Need at least 40 GiB free on the compute-node temporary disk')
PY
    LONGEVAL_NODE_STAGE="$(mktemp -d "$LONGEVAL_TEMP_ROOT/longeval-${SLURM_JOB_ID}.XXXXXX")"
fi
LONGEVAL_NODE_STAGE="$(realpath -e "$LONGEVAL_NODE_STAGE")"
case "$LONGEVAL_NODE_STAGE/" in
    "$LONGEVAL_TEMP_ROOT/"?*/) ;;
    *) echo 'Encoder stage must be inside the node temporary directory' >&2; return 1 ;;
esac
export LONGEVAL_NODE_STAGE LONGEVAL_NODE_TMP_ROOT="$LONGEVAL_TEMP_ROOT"
export TMPDIR="$LONGEVAL_NODE_STAGE/tmp" TMP="$LONGEVAL_NODE_STAGE/tmp" TEMP="$LONGEVAL_NODE_STAGE/tmp"
export PIP_CACHE_DIR="$LONGEVAL_NODE_STAGE/pip-cache" UV_CACHE_DIR="$LONGEVAL_NODE_STAGE/uv-cache"
export UV_PYTHON_INSTALL_DIR="$LONGEVAL_NODE_STAGE/python" UV_PYTHON_BIN_DIR="$LONGEVAL_NODE_STAGE/bin"
export XDG_CACHE_HOME="$LONGEVAL_NODE_STAGE/cache" HF_HOME="$LONGEVAL_NODE_STAGE/huggingface"
export HF_HUB_CACHE="$LONGEVAL_NODE_STAGE/huggingface/hub" HUGGINGFACE_HUB_CACHE="$LONGEVAL_NODE_STAGE/huggingface/hub"
export TORCH_HOME="$LONGEVAL_NODE_STAGE/torch" CUDA_CACHE_PATH="$LONGEVAL_NODE_STAGE/cuda-cache"
export TRITON_CACHE_DIR="$LONGEVAL_NODE_STAGE/triton-cache" PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_CACHE_DIR="$LONGEVAL_NODE_STAGE/inductor-cache" PIP_CONFIG_FILE=/dev/null
mkdir -p "$TMPDIR"
