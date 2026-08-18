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
module load Miniforge3

: "${DT_ENV:?DT_ENV is required}"
: "${IM2NAV_ROOT:?IM2NAV_ROOT is required}"
: "${V1_FROZEN_DIR:?V1_FROZEN_DIR is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

source activate "$DT_ENV"

cd "$SLURM_SUBMIT_DIR"

export PYTHONUNBUFFERED=1

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

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Twin V2 full LOSO"
echo "SLURM job:       $SLURM_JOB_ID"
echo "Array job:       $SLURM_ARRAY_JOB_ID"
echo "Array task:      $SLURM_ARRAY_TASK_ID"
echo "Sequence:        $SEQUENCE"
echo "Base seed:       $BASE_SEED"
echo "Host:            $(hostname)"
echo "CUDA devices:    ${CUDA_VISIBLE_DEVICES:-unset}"
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

echo
echo "============================================================"
echo "RUN COMPLETE"
echo "Sequence: $SEQUENCE"
echo "Base seed: $BASE_SEED"
echo "============================================================"