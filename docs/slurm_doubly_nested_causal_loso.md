# Causal Doubly Nested LOSO on Falcon

This run repeats the full doubly nested trajectory-generation study with
backward-only derivatives in **both V1 and V2**. It is separate from the
existing centered-gradient bank and writes to
`results/doubly_nested_causal_loso_slurm/`.

The full design has 300 V1-to-V2 pipelines (10 outer folds, 10 targets per
outer fold, and three seeds), or up to 600 model fits. It uses the same
outer-test and qualification-training exclusions as the centered study.
Every split, derivative mode, checkpoint, V1 results source, and trajectory
hash is audited. A task is resumed only when its artifacts are complete and
their recorded derivative mode matches the requested mode.

## 1. Update the checkout

On Falcon, pull the committed study changes and confirm the checkout is clean:

```bash
git pull
git status --short
```

The submission helper refuses tracked, uncommitted changes so the queued job
is pinned to one repository commit.

## 2. Preflight without training

Run this from the repository root. It validates paths and runner options and
prints the causal-mode commands and runs the numerical no-lookahead preflight;
it does not train models:

```bash
python -m DigitalTwin.analysis.run_doubly_nested_loso \
  --feature-derivative-mode causal_backward \
  --dry-run --start-task 2 --stop-task 4 \
  --output-dir results/doubly_nested_causal_loso_preflight
```

Inspect `results/doubly_nested_causal_loso_preflight/dry_run_commands.json` and
confirm that **both** V1 and V2 commands include
`--feature-derivative-mode causal_backward` and that the qualification-training
task excludes its outer sequence.

## 3. Python/GPU environment

Use the same CUDA-enabled environment as the centered run. For example, after
activating the environment on Falcon:

```bash
python -c "import numpy, pandas, scipy, sklearn, matplotlib, torch"
srun --account=ceisca-allocation \
  --partition=a30_normal_q --qos=fal_a30_normal_base --gres=gpu:1 \
  --time=00:05:00 "$(which python)" -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Use the Python path that successfully reports CUDA availability below.
If plotting/analysis imports fail, install those dependencies in the activated
environment before submission:

```bash
python -m pip install matplotlib scikit-learn scipy
```

## 4. Smoke task, then full run

First submit two causal tasks into their own smoke directory: one outer-test
pipeline and one qualification-training pipeline that must exclude the outer
sequence. This is four model fits total. It is intentionally not a complete
evidence bank, so the post-run report will say it is partial and will not run
the paper-level evidence analysis:

```bash
OUTPUT_DIR="$PWD/results/doubly_nested_causal_loso_smoke" \
RUN_START_TASK=2 RUN_STOP_TASK=4 \
PYTHON_BIN="$(which python)" \
bash scripts/slurm/submit_doubly_nested_causal_loso.sh \
  --account=ceisca-allocation \
  --partition=a30_normal_q --qos=fal_a30_normal_base \
  --time=08:00:00
```

Check the task log and audit. Confirm `feature_derivative_mode` is
`causal_backward` in `run_config.json`, the V1 `fold_splits.json`, and the V2
`run_manifest.json`. Confirm
`causal_no_lookahead_preflight.json` reports `passed: true`; it checks V1 and
V2 derivative features, the first-sample boundary rule, and V2 trailing sensor
windows. The two smoke tasks exercise both the outer-test and
qualification-training paths.

Then submit the full 300-task study (omit smoke overrides):

```bash
PYTHON_BIN="$(which python)" \
bash scripts/slurm/submit_doubly_nested_causal_loso.sh \
  --account=ceisca-allocation \
  --partition=a30_normal_q --qos=fal_a30_normal_base
```

If your environment is loaded from a setup file inside the batch job, use:

```bash
ENV_SETUP="$PWD/scripts/slurm/falcon_env.sh" PYTHON_BIN=python \
bash scripts/slurm/submit_doubly_nested_causal_loso.sh \
  --account=ceisca-allocation \
  --partition=a30_normal_q --qos=fal_a30_normal_base
```

## 5. Monitor and resume

```bash
squeue -u "$USER"
tail -f slurm-dt-causal-nested-<JOBID>.out
tail -n 80 slurm-dt-causal-nested-<JOBID>.err
```

If the allocation ends before completion, resubmit the same command with the
same output directory at the same repository commit. Incomplete V2 task
directories are rebuilt using the model runner's overwrite option. Completed
causal tasks are validated and skipped; the
centered bank is never used as a checkpoint source. Do not point this study at
`results/doubly_nested_loso_slurm/`.

The job runs `service_evidence_from_bank` automatically only after the manifest
reports a complete, clean 300-trajectory bank. To rerun that analysis manually:

```bash
python -m DigitalTwin.analysis.service_evidence_from_bank \
  --bank-manifest results/doubly_nested_causal_loso_slurm/merged_manifest.json \
  --output results/doubly_nested_causal_loso_slurm/service_evidence \
  --expected-feature-derivative-mode causal_backward
```

Completion checks:

```text
results/doubly_nested_causal_loso_slurm/full_audit.json:
  complete = true
  verified_tasks = 300
results/doubly_nested_causal_loso_slurm/merged_manifest.json:
  ready_for_evidence_analysis = true
  feature_derivative_mode = causal_backward
  causal_no_lookahead_preflight.passed = true
results/doubly_nested_causal_loso_slurm/service_evidence/evidence_manifest.json:
  status = complete
```

## Timing interpretation

`causal_backward` computes derivative features from the current and earlier
prepared samples, eliminating the next-sample lookahead used by centered
differences. The source sensor resampling, initial calibration, and reference
alignment follow the existing offline preparation protocol. The numerical
preflight covers derivative features and trailing windows on those prepared
signals.
The replay treats a causally computed state as available at its source-grid
timestamp before configured network delay. Edge inference/computation latency
is not measured by this experiment; do not describe this run as a measured
end-to-end compute-latency study.

The output is a new causal-feature nested result. It can be compared with the
centered bank for source fidelity, held-out qualification risk, and ledger
counts. It does not replace or overwrite the centered evidence.
