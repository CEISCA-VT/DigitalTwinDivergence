#!/bin/bash
#SBATCH --job-name=dt-v2-loso
#SBATCH --partition=a30_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-29%10
#SBATCH --output=slurm-v2-%A-%a.out
#SBATCH --error=slurm-v2-%A-%a.err

set -euo pipefail

module reset
module load Miniconda3/25.11.1-1

REPO_ROOT="$SLURM_SUBMIT_DIR"
DT_ENV="$HOME/envs/digitaltwin-a30"
IM2NAV_ROOT="$REPO_ROOT/public_datasets/im2nav"
V1_FROZEN_DIR="$REPO_ROOT/results/i2nav_v1_frozen"
OUTPUT_DIR="/scratch/$USER/i2nav_v2_full_loso"

cd "$REPO_ROOT"
source activate "$DT_ENV"

export PYTHONUNBUFFERED=1

test -d "$IM2NAV_ROOT"
test -f "$V1_FROZEN_DIR/FROZEN_MANIFEST.json"
mkdir -p "$OUTPUT_DIR"

SEQUENCES=(
    building00
    building01
    building02
    parking00
    parking01
    parking02
    playground00
    street00
    street01
    street02
)

BASE_SEEDS=(
    42
    1042
    2042
)

REPLICATE_INDEX=$((SLURM_ARRAY_TASK_ID / 10))
FOLD_INDEX=$((SLURM_ARRAY_TASK_ID % 10))

SEQUENCE="${SEQUENCES[$FOLD_INDEX]}"
BASE_SEED="${BASE_SEEDS[$REPLICATE_INDEX]}"

echo "============================================================"
echo "Twin V2 full LOSO"
echo "SLURM job:       $SLURM_JOB_ID"
echo "Array job:       $SLURM_ARRAY_JOB_ID"
echo "Array task:      $SLURM_ARRAY_TASK_ID"
echo "Sequence:        $SEQUENCE"
echo "Base seed:       $BASE_SEED"
echo "Host:            $(hostname)"
echo "CUDA devices:    ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repository:      $REPO_ROOT"
echo "Git commit:      $(git rev-parse HEAD)"
echo "i2Nav root:      $IM2NAV_ROOT"
echo "Frozen V1:       $V1_FROZEN_DIR"
echo "Output:          $OUTPUT_DIR"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m DigitalTwin.analysis.i2nav_v2_full_loso \
    --root "$IM2NAV_ROOT" \
    --frozen-v1-dir "$V1_FROZEN_DIR" \
    --test-sequence "$SEQUENCE" \
    --base-seed "$BASE_SEED" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda

echo "============================================================"
echo "RUN COMPLETE"
echo "Sequence: $SEQUENCE"
echo "Base seed: $BASE_SEED"
echo "============================================================"
