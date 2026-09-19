#!/usr/bin/env bash
#SBATCH --job-name=dt-nested-loso
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_DIR}" || ! -d "${REPO_DIR}/DigitalTwin" ]]; then
    echo "REPO_DIR must point to the DigitalTwinDivergence repository." >&2
    exit 2
fi

cd "${REPO_DIR}"

ENV_SETUP="${ENV_SETUP:-}"
if [[ -n "${ENV_SETUP}" ]]; then
    if [[ ! -f "${ENV_SETUP}" ]]; then
        echo "ENV_SETUP was provided but does not exist: ${ENV_SETUP}" >&2
        exit 2
    fi
    # shellcheck disable=SC1090
    source "${ENV_SETUP}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_DIR}/public_datasets/im2nav}"
FROZEN_V1_DIR="${FROZEN_V1_DIR:-${REPO_DIR}/results/i2nav_v1_frozen}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results/doubly_nested_loso_slurm}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-$(git rev-parse HEAD)}"

echo "Repository: ${REPO_DIR}"
echo "Commit: ${EXPECTED_COMMIT}"
echo "Python: ${PYTHON_BIN}"
echo "Dataset: ${DATASET_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "Workers: ${NUM_WORKERS}"
echo "ENV_SETUP: ${ENV_SETUP:-<none>}"
command -v "${PYTHON_BIN}" || true
"${PYTHON_BIN}" -c "import sys; print('python_executable', sys.executable); print('python_version', sys.version)"
nvidia-smi || true

"${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('torch', torch.__version__, 'GPU', torch.cuda.get_device_name(0))"

srun "${PYTHON_BIN}" -m DigitalTwin.analysis.run_doubly_nested_loso \
    --dataset-root "${DATASET_ROOT}" \
    --frozen-v1-dir "${FROZEN_V1_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --num-workers "${NUM_WORKERS}" \
    --python "${PYTHON_BIN}" \
    --expected-commit "${EXPECTED_COMMIT}"

"${PYTHON_BIN}" -m DigitalTwin.analysis.service_evidence_from_bank \
    --bank-manifest "${OUTPUT_DIR}/merged_manifest.json" \
    --output "${OUTPUT_DIR}/service_evidence"

echo "Doubly nested LOSO and corrected service-evidence analysis completed."
