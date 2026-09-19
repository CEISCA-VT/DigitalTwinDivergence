# Doubly Nested LOSO on One Slurm Job

The repository-native driver runs the complete study as one resumable shard:

- 10 outer physical sequences;
- 3 seeds;
- 1 outer-test trajectory plus 9 qualification-training trajectories per
  outer sequence;
- 300 V1-to-V2 pipelines, or up to 600 model fits.

No predictions are ensembled. Each trajectory remains an independent frozen
fold/seed result.

## Preflight on the login node

From a clean committed repository:

```bash
python -m DigitalTwin.analysis.run_doubly_nested_loso \
  --dry-run \
  --max-tasks 2 \
  --device cuda \
  --output-dir results/doubly_nested_loso_slurm_preflight
```

The preflight validates all ten dataset directories, the frozen V1 manifest,
runner options, the 300-task graph, repository state, and generated commands.
It performs no training.

## Submit

The default request is one GPU, eight CPUs, 64 GB RAM, and seven days. Override
partition, account, time, memory, or GPU syntax on the `sbatch` command if your
cluster requires different values.

## Python environment on Falcon

The Slurm job must run with a Python environment that can import CUDA-enabled
PyTorch. A failed job with:

```text
ModuleNotFoundError: No module named 'torch'
```

means the batch job used the system Python instead of the project environment.

First inspect what Falcon exposes:

```bash
module spider python
module spider conda
module spider miniforge
module spider mamba
```

If a conda/miniforge module exists, create the environment once from the repo:

```bash
module load <the-python-or-conda-module-shown-by-module-spider>
conda env create -f environment.yml
conda activate digitaltwin-a30
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then submit with the exact Python from that environment:

```bash
PYTHON_BIN="$(which python)" bash scripts/slurm/submit_doubly_nested_loso.sh --partition=a30_normal_q --qos=fal_a30_normal_base
```

If conda is not available, create a local virtual environment once:

```bash
python3 -m venv "$HOME/.venvs/digitaltwin-a30"
source "$HOME/.venvs/digitaltwin-a30/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --index-url https://download.pytorch.org/whl/cu118 "torch==2.5.1"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then submit with:

```bash
PYTHON_BIN="$HOME/.venvs/digitaltwin-a30/bin/python" bash scripts/slurm/submit_doubly_nested_loso.sh --partition=a30_normal_q --qos=fal_a30_normal_base
```

The batch script also supports an optional setup file if your cluster requires
module loads inside the batch allocation:

```bash
cat > scripts/slurm/falcon_env.sh <<'EOF'
module load <module-name-if-needed>
source "$HOME/.venvs/digitaltwin-a30/bin/activate"
EOF

ENV_SETUP="$PWD/scripts/slurm/falcon_env.sh" PYTHON_BIN=python bash scripts/slurm/submit_doubly_nested_loso.sh --partition=a30_normal_q --qos=fal_a30_normal_base
```

Do a cheap GPU import check before submitting the week-scale job:

```bash
srun --partition=a30_normal_q --qos=fal_a30_normal_base --gres=gpu:1 --time=00:05:00 "$HOME/.venvs/digitaltwin-a30/bin/python" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Use the submission helper so the queued job is pinned to the exact current
commit:

```bash
bash scripts/slurm/submit_doubly_nested_loso.sh
```

Scheduler options can be forwarded, for example:

```bash
bash scripts/slurm/submit_doubly_nested_loso.sh --partition=gpu --account=my_account
```

To select a specific Python environment:

```bash
PYTHON_BIN=/path/to/env/bin/python bash scripts/slurm/submit_doubly_nested_loso.sh
```

Optional exported paths are `REPO_DIR`, `DATASET_ROOT`, `FROZEN_V1_DIR`, and
`OUTPUT_DIR`. The default repository is `$SLURM_SUBMIT_DIR`, so submit from the
repository root.

## Resume after timeout or interruption

Submit the same command again with the same `OUTPUT_DIR`. A V1 stage is skipped
only when its checkpoint, results CSV, and split manifest are structurally
complete. A V2 stage is skipped only when its completion marker, model,
trajectory, trace, profile, time series, and exact V1 source paths are present.
Every newly completed task is audited and appended atomically to
`task_status.json`.

The job is deliberately fail-fast. Inspect:

```text
results/doubly_nested_loso_slurm/logs/<task-id>.log
```

then correct the environment or storage problem and resubmit. Do not delete
completed task directories.

## Completion criteria

The run is complete only when:

```text
full_audit.json: complete = true
merged_manifest.json: ready_for_evidence_analysis = true
verified_trajectories = 300
```

The Slurm script then runs the corrected qualification-surface, comparator,
risk-coverage, and remediability analysis automatically into:

```text
results/doubly_nested_loso_slurm/service_evidence/
```

If the scheduler terminates the job after task 300 but before post-analysis,
run:

```bash
python -m DigitalTwin.analysis.service_evidence_from_bank \
  --bank-manifest results/doubly_nested_loso_slurm/merged_manifest.json \
  --output results/doubly_nested_loso_slurm/service_evidence
```

## Resource warning

One shard does not mean one training run. It means 300 sequential pipelines.
The seven-day request is a starting value, not a guaranteed runtime. Check your
cluster's maximum wall time and available storage before submission. The run is
safe to resume if it spans multiple allocations.
