#!/usr/bin/env bash
# Select partition/account/CPU constraint after running longeval-preflight.sh.
# Scientific timing runs reserve one node; each adapter and BLAS uses one thread.
#SBATCH --job-name=longeval-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=longeval-cpu-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
: "${LONGEVAL_PYTHON:?set the Python path in the prepared research environment}"
: "${LONGEVAL_OUTPUT:?set a new output directory for this job}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
hostname
lscpu
git rev-parse HEAD
"$LONGEVAL_PYTHON" -m pip freeze
case "${LONGEVAL_MODE:-pilot}" in
    pilot)
        : "${LONGEVAL_DATASET:?set the fixture dataset.json path}"
        command=("$LONGEVAL_PYTHON" -m research.longeval run --dataset "$LONGEVAL_DATASET" --output "$LONGEVAL_OUTPUT" --native-origin checkout-build)
        ;;
    audit)
        : "${LONGEVAL_DATA:?set the folder containing all three verified extractions}"
        command=("$LONGEVAL_PYTHON" -m research.longeval audit-vectors "$LONGEVAL_DATA" --output "$LONGEVAL_OUTPUT")
        ;;
    feasibility)
        : "${LONGEVAL_DATA:?set one verified snapshot extraction directory}"
        : "${LONGEVAL_AUDIT:?set the completed scan.json for that snapshot}"
        : "${LONGEVAL_BACKEND:?select epzero, faiss_hnsw or faiss_ivf}"
        command=("$LONGEVAL_PYTHON" -m research.longeval feasibility "$LONGEVAL_DATA" --audit "$LONGEVAL_AUDIT" --output "$LONGEVAL_OUTPUT" --snapshot "${LONGEVAL_SNAPSHOT:-1}" --backend "$LONGEVAL_BACKEND" --count "${LONGEVAL_COUNT:-64000}" --native-origin checkout-build)
        ;;
    *) printf 'Unknown LONGEVAL_MODE\n' >&2; exit 2 ;;
esac
srun --cpu-bind=cores --mem-bind=local "${command[@]}"
