#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked files are modified. Commit the study code before submission." >&2
    exit 2
fi

EXPECTED_COMMIT="$(git rev-parse HEAD)"
echo "Submitting commit ${EXPECTED_COMMIT}"

sbatch \
    --export="ALL,REPO_DIR=${REPO_DIR},EXPECTED_COMMIT=${EXPECTED_COMMIT}" \
    "$@" \
    scripts/slurm/doubly_nested_loso.sbatch
