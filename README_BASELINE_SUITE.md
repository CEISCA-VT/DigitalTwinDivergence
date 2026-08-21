# i2Nav External Baseline + Fidelity Evaluation Suite

This bundle adds a reproducible baseline study for the sensor-lightweight mobile-robot digital-twin paper without changing Twin V2.

## Where the three ZIPs go

1. Extract `extract_to_DigitalTwin_baselines.zip` into:

   `DigitalTwin/baselines/`

2. Extract `extract_to_DigitalTwin_analysis.zip` into:

   `DigitalTwin/analysis/`

3. Extract `extract_to_repo_root.zip` into the repository root, next to `DigitalTwin/`, `results/`, `README.md`, etc.

The ZIPs are intentionally separated by destination so no guessing is required.

## What is implemented

### Maintenance baselines

- **Wheel-IMU EKF (planar EKF-IW)**: classical planar wheel/IMU EKF compatible with the channels frozen in the V2 archive. It estimates gyro bias using wheel-derived yaw rate when available and uses only training-fold calibration + the held-out initial pose.
- **LWOI-IMU adaptation (sparse RBF residual)**: strict 9-sequence LOSO residual learning inspired by Brossard & Bonnabel's learned wheel/IMU model-error correction. It is deliberately labelled an adaptation, not an exact reproduction of the authors' original FoG/Pyro-GP setup.
- **YNet-style reduced-input yaw + EKF**: causal temporal-convolution + attention yaw-rate estimator trained LOSO and fused in the planar EKF. The frozen CSVs do not contain the full raw multimodal sensor packet set used by the published YNet, so this is also deliberately labelled a reduced-input adaptation.
- **Fixed Physics (recomputed)** is available as a sanity baseline but is **not run by default**. For the paper's official Fixed-Physics comparison, use your already-frozen Fixed Physics trajectories.
- **WING is not reimplemented**. `external_adapter.py` is provided so a future official/validated WING output can be normalized into the same evaluation schema without pretending an unvalidated reimplementation is WING.

### Fidelity/evaluator baselines

- **TFP multi-method evaluator**: ATE, heading MAE, RPE1/5/10, instantaneous position/heading tails, pose-derived speed/yaw component divergence, signed heading/yaw residuals, and accumulated yaw-residual diagnostics.
- **Bergs-style trajectory evaluation**: symmetric Hausdorff distance, mean bidirectional nearest-path distance, terminal position/heading error, and path-length mismatch.
- **Muñoz-style trace alignment**: position and heading evaluated separately, affine gaps, 1 Hz default sampling, MAD sensitivity grids. This remains a paper-faithful adaptation, **not the authors' official artifact**.
- **Framework comparison**: rank correlations, local-good/global-bad cases, parking02 diagnostics, and method-level tables. It intentionally does **not** create a scalar evaluator “winner.”
- **Protocol validation**: checks trajectory schemas, monotonic timestamps, LOSO leakage, failed jobs, and reproduces the frozen V2 headline metrics within tolerance when the real archive is used.

## One required edit before the Fixed Physics Muñoz comparison

Open `baseline_suite_config.json`.

The V2 block is already configured for:

`results/i2nav_v2_full_loso/i2nav_v2_full_loso`

For your **frozen official Fixed Physics trajectories**, change this block:

```json
{
  "name": "Fixed Physics",
  "enabled": false,
  "root": "results/PUT_FROZEN_FIXED_PHYSICS_ROOT_HERE",
  "glob": "**/*evaluated_trajectory.csv",
  "required": false,
  "expected_files": 10
}
```

to the real root and set `"enabled": true`.

Do **not** enable the V1 block for an official comparison unless you regenerate equivalent V1 trajectories under the same frozen protocol. The existing internal V1→V2 numbers can remain an ablation/evolution result.

## First run: smoke test

From the repository root in PowerShell:

```powershell
.\run_i2nav_baseline_suite.ps1 -Smoke
```

This runs EKF-IW, LWOI-IMU, and YNet-style on two held-out sequences with tiny learned-model settings, then runs synthetic Muñoz checks. It intentionally stops before the expensive full evaluation.

If PowerShell blocks local scripts:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_i2nav_baseline_suite.ps1 -Smoke
```

## Full run

```powershell
.\run_i2nav_baseline_suite.ps1
```

Main generated directories:

- `results/i2nav_external_baselines/` — generated maintenance-baseline trajectories/checkpoints
- `results/i2nav_fidelity_baselines/tfp/`
- `results/i2nav_fidelity_baselines/bergs/`
- `results/i2nav_fidelity_baselines/munoz/`
- `results/i2nav_fidelity_baselines/comparison/`
- `results/i2nav_fidelity_baselines/validation/`

Muñoz is the slowest evaluation because affine trace alignment is O(NM). The script supports `--resume`, and the PowerShell runner enables it automatically.

## Useful partial runs

Run only EKF-IW first:

```powershell
python -m DigitalTwin.baselines.run_i2nav_baselines `
  --input-root "results\i2nav_v2_full_loso\i2nav_v2_full_loso" `
  --models "ekf_iw"
```

Run EKF-IW + LWOI but skip the slower neural YNet-style adaptation:

```powershell
.\run_i2nav_baseline_suite.ps1 -SkipYNet
```

Reuse already-generated baseline trajectories without retraining:

```powershell
.\run_i2nav_baseline_suite.ps1 -SkipTraining
```

Run TFP + Bergs and skip the expensive Muñoz stage:

```powershell
.\run_i2nav_baseline_suite.ps1 -SkipTraining -SkipMunoz
```

## External WING / official LWOI / official YNet later

If you obtain a validated external trajectory, normalize it with:

```powershell
python -m DigitalTwin.baselines.external_adapter `
  --input "PATH_TO_EXTERNAL_OUTPUT.csv" `
  --reference "PATH_TO_MATCHING_I2NAV_REFERENCE.csv" `
  --method-name "WING" `
  --output "results\external\WING\parking02\evaluated_trajectory.csv" `
  --column-map-json '{"time_s":"timestamp","estimate_east_m":"x","estimate_north_m":"y","estimate_heading_rad":"yaw"}'
```

Then add that output set as another enabled `official_methods`/external method entry in `baseline_suite_config.json` and rebuild the fidelity manifest.

## Publication-safe naming

Use the following names unless/until the adaptations are validated against the original implementations:

- `Wheel–IMU EKF (planar EKF-IW)`
- `LWOI-IMU adaptation (sparse RBF residual)`
- `YNet-style reduced-input yaw + EKF`
- `Muñoz-style trace-alignment adaptation`

Do **not** call the last three “exact LWOI,” “exact YNet,” or “official Muñoz implementation.” WING is intentionally adapter-only in this bundle.

## Expected V2 freeze check

On the real frozen V2 archive the final validation expects approximately:

- ATE: 2.398 m
- heading MAE: 2.569 deg
- RPE1: 0.0611 m
- RPE5: 0.1603 m
- RPE10: 0.2532 m

If those fail by more than 1%, stop and inspect the source path/schema before interpreting any baseline result.
