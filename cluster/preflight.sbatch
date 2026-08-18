#!/bin/bash
#SBATCH --job-name=dt-v2-preflight
#SBATCH --partition=a30_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-preflight-%j.out
#SBATCH --error=slurm-preflight-%j.err

set -euo pipefail

module reset
module load Miniforge3

: "${DT_ENV:?DT_ENV is required}"
: "${IM2NAV_ROOT:?IM2NAV_ROOT is required}"
: "${V1_FROZEN_DIR:?V1_FROZEN_DIR is required}"

source activate "$DT_ENV"

cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "Digital Twin V2 HPC preflight"
echo "============================================================"
echo "Host: $(hostname)"
echo "Repo: $SLURM_SUBMIT_DIR"
echo "Environment: $DT_ENV"
echo "i2Nav: $IM2NAV_ROOT"
echo "Frozen V1: $V1_FROZEN_DIR"
echo

test -d "$IM2NAV_ROOT"
test -d "$V1_FROZEN_DIR"
test -f "$V1_FROZEN_DIR/FROZEN_MANIFEST.json"

echo "Dataset/frozen-V1 paths: PASS"

python --version

python -c "import torch; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0)); print('capability:', torch.cuda.get_device_capability(0))"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

python -m py_compile DigitalTwin/analysis/i2nav_v2_full_loso.py DigitalTwin/analysis/i2nav_fidelity_evaluator.py

python -c "import DigitalTwin.analysis.i2nav_loso_ablation; import DigitalTwin.analysis.i2nav_v2_yaw_bias_pilot; import DigitalTwin.analysis.i2nav_v2_full_loso; import DigitalTwin.analysis.i2nav_fidelity_evaluator; print('DigitalTwin imports: PASS')"

python -m DigitalTwin.analysis.i2nav_v2_full_loso --help > /dev/null
python -m DigitalTwin.analysis.i2nav_fidelity_evaluator --help > /dev/null

echo
echo "============================================================"
echo "PRE-FLIGHT PASS"
echo "============================================================"