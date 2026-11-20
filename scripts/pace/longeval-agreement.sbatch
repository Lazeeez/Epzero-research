#!/usr/bin/env bash
#SBATCH --job-name=longeval-agreement
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu,ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --tmp=80G
#SBATCH --time=04:00:00
#SBATCH --output=longeval-agreement-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
REFERENCE="$ROOT/build/longeval-gpu-inputs/snapshot-1"
OUTPUT="$ROOT/build/pace-longeval-encoder/jobs/${SLURM_JOB_ID:?run inside a Slurm allocation}"
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    if [ -n "${STAGE:-}" ]; then
        for name in packages.txt torch-install.json encoder-install.json; do
            if [ -f "$STAGE/$name" ]; then cp "$STAGE/$name" "$OUTPUT/" || true; fi
        done
        if [ -f "$STAGE/model/model-identity.json" ]; then
            cp "$STAGE/model/model-identity.json" "$OUTPUT/" || true
        fi
    fi
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    exit "$code"
}
trap finish EXIT
source scripts/pace/longeval-node-env.sh
STAGE="$LONGEVAL_NODE_STAGE"
PYTHON="$STAGE/venv/bin/python"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
"$ROOT/build/pace-longeval/venv/bin/python" scripts/pace/longeval-gpu-verify.py
cp longeval-gpu-manifest.json "$OUTPUT/"
cp "$REFERENCE/reference.json" "$OUTPUT/"
{
    printf 'node_stage=%s\n' "$STAGE"
    findmnt -T "$STAGE" -o SOURCE,FSTYPE,TARGET
    df -h "$STAGE"
} > "$OUTPUT/node-storage.txt"
# Package and model downloads happen in this allocation, directly to its temporary disk.
bash scripts/pace/longeval-gpu-setup.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
{
    date -u
    hostname
    nvidia-smi
    "$PYTHON" -m pip freeze
} > "$OUTPUT/hardware-and-packages.txt"
srun "$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError('Requires a working CUDA BF16 GPU')
if torch.cuda.get_device_properties(0).total_memory < 32 * 1024**3:
    raise RuntimeError('Requires at least 32 GiB VRAM for the float32 diagnostic')
print('GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
PY
# Each precision gets a separate process and output; neither changes the reference vectors.
for precision in bfloat16 float32; do
    srun "$PYTHON" -m research.longeval encode-documents "$STAGE/model" "$REFERENCE/records.jsonl" \
        --output "$OUTPUT/$precision" --dtype "$precision" --device cuda --threads 4 --max-length 8192
    "$PYTHON" -m research.longeval.representation compare "$REFERENCE" "$OUTPUT/$precision" "$OUTPUT/$precision-agreement.json"
done
"$PYTHON" scripts/pace/longeval-gpu-verify.py
printf 'PASS: both precision diagnostics completed; inspect agreement, not just this exit status: %s\n' "$OUTPUT"
