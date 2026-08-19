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
module load Miniconda3/25.11.1-1

REPO_ROOT="$SLURM_SUBMIT_DIR"
DT_ENV="$HOME/envs/digitaltwin-a30"
IM2NAV_ROOT="$REPO_ROOT/public_datasets/im2nav"
V1_FROZEN_DIR="$REPO_ROOT/results/i2nav_v1_frozen"

cd "$REPO_ROOT"
source activate "$DT_ENV"

echo "============================================================"
echo "Digital Twin V2 Falcon preflight"
echo "============================================================"
echo "Host:            $(hostname)"
echo "Repository:      $REPO_ROOT"
echo "Conda env:       $DT_ENV"
echo "i2Nav root:      $IM2NAV_ROOT"
echo "Frozen V1:       $V1_FROZEN_DIR"
echo "CUDA devices:    ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Git commit:      $(git rev-parse HEAD)"
echo "============================================================"

test -d "$IM2NAV_ROOT"
test -d "$V1_FROZEN_DIR"
test -f "$V1_FROZEN_DIR/FROZEN_MANIFEST.json"

for seq in building00 building01 building02 parking00 parking01 parking02 playground00 street00 street01 street02; do
    test -d "$IM2NAV_ROOT/$seq"
done

V1_CHECKPOINTS=$(find "$V1_FROZEN_DIR/checkpoints" -type f -name "*.pt" | wc -l)
echo "Frozen V1 checkpoints found: $V1_CHECKPOINTS"
if [ "$V1_CHECKPOINTS" -lt 30 ]; then
    echo "ERROR: Expected at least 30 frozen V1 .pt checkpoints."
    exit 1
fi

python --version

python -c "import torch; print('Torch:', torch.__version__); print('Torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is not available'; print('GPU:', torch.cuda.get_device_name(0)); print('Capability:', torch.cuda.get_device_capability(0))"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

python -m py_compile DigitalTwin/analysis/i2nav_v2_full_loso.py DigitalTwin/analysis/i2nav_fidelity_evaluator.py

python -c "import DigitalTwin.analysis.i2nav_loso_ablation; import DigitalTwin.analysis.i2nav_v2_yaw_bias_pilot; import DigitalTwin.analysis.i2nav_v2_full_loso; import DigitalTwin.analysis.i2nav_fidelity_evaluator; print('DigitalTwin imports: PASS')"

python -m DigitalTwin.analysis.i2nav_v2_full_loso --help >/dev/null
python -m DigitalTwin.analysis.i2nav_fidelity_evaluator --help >/dev/null

echo "============================================================"
echo "PRE-FLIGHT PASS"
echo "============================================================"
