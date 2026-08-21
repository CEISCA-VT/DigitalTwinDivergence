# Fixed Physics / EKF-IW repair patch

Extract at the repository root.

## What was actually wrong

The previous planar EKF treated `abs(odo_speed_mps) <= threshold` as a stationary condition and injected a zero-yaw-rate pseudo measurement. That assumption is invalid for a differential/skid-steer mobile robot because it can rotate in place or execute a low-forward-speed turn. The pseudo update can reinterpret real rotation as gyro bias and destroy global heading.

The corrected EKF:
- removes the `v≈0 => yaw rate = 0` update entirely;
- learns yaw-rate sign/scale/bias from training sequences only (strict LOSO);
- checks whether the wheel-derived yaw channel is informative before using it;
- uses an innovation gate for wheel-yaw bias updates;
- falls back to calibrated IMU propagation if wheel yaw is not trustworthy;
- records update/calibration diagnostics in every output.

## Fixed Physics clarification

The large `Fixed Physics (recomputed)` ATE in TFP is **un-aligned synchronized operational error**. The frozen official i2Nav Fixed Physics value (3.299 m) is an **aligned official APE**. They are different metrics. The included validator first checks the same alignment implementation against frozen V2 (target 1.635 m), then tests whether recomputed Fixed Physics approaches the 3.299 m official magnitude.

## Run

First run a quick repair smoke test:

```powershell
.\repair_fixed_ekf_baselines.ps1 -Smoke
```

If the EKF diagnostics look reasonable, run the full deterministic repair and reevaluation:

```powershell
.\repair_fixed_ekf_baselines.ps1
```

This preserves the already-trained LWOI/YNet files and their manifest entries. It rewrites only Fixed Physics and EKF trajectories, then recomputes TFP/Bergs/Muñoz so no stale evaluator result remains.

If Muñoz is too slow during the first check:

```powershell
.\repair_fixed_ekf_baselines.ps1 -SkipMunoz
```

Then run the full command once the deterministic metrics look correct.

## Send back

Please send:
- `results/i2nav_fidelity_baselines/validation/ekf_iw_diagnostics.csv`
- `results/i2nav_fidelity_baselines/validation/official_alignment_validation_summary.csv`
- updated `results/i2nav_fidelity_baselines/tfp/tfp_dataset_summary.csv`
- updated `results/i2nav_fidelity_baselines/tfp/tfp_pairwise_vs_reference.csv`
